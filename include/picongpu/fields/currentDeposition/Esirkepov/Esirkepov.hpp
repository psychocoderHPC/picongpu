/* Copyright 2013-2018 Axel Huebl, Heiko Burau, Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"
#include "picongpu/algorithms/Set.hpp"
#include <pmacc/types.hpp>
#include <pmacc/cuSTL/cursor/Cursor.hpp>
#include <pmacc/cuSTL/cursor/tools/twistVectorFieldAxes.hpp>
#include <pmacc/cuSTL/cursor/compile-time/SafeCursor.hpp>
#include <pmacc/nvidia/atomic.hpp>

#include "picongpu/fields/currentDeposition/Esirkepov/Esirkepov.def"
#include "picongpu/fields/currentDeposition/Esirkepov/Line.hpp"
#include "picongpu/fields/currentDeposition/RelayPoint.hpp"

#include "pmacc/nvidia/warp.hpp"
#include <pmacc/memory/boxes/SharedBox.hpp>
#include <pmacc/mappings/threads/ThreadCollective.hpp>

#include <mma.h>

#define PICONGPU_NUMWARPS 4
#define PIC_ENFORCE_SHARED_ATOMICS 1
#define PIC_USE_MMA 1

namespace picongpu
{
namespace currentSolver
{
using namespace pmacc;

template<typename T_ParticleShape>
struct Esirkepov<T_ParticleShape, DIM3>
{
    using ParticleAssign = typename T_ParticleShape::ChargeAssignment;
    static constexpr int supp = ParticleAssign::support;

    static constexpr int currentLowerMargin = supp / 2 + 1 - (supp + 1) % 2;
    static constexpr int currentUpperMargin = (supp + 1) / 2 + 1;
    typedef pmacc::math::CT::Int<currentLowerMargin, currentLowerMargin, currentLowerMargin> LowerMargin;
    typedef pmacc::math::CT::Int<currentUpperMargin, currentUpperMargin, currentUpperMargin> UpperMargin;

    PMACC_CASSERT_MSG(
        __Esirkepov_supercell_or_number_of_guard_supercells_is_too_small_for_stencil,
        pmacc::math::CT::min<
            typename pmacc::math::CT::mul<
                SuperCellSize,
                GuardSize
            >::type
        >::type::value >= currentLowerMargin &&
        pmacc::math::CT::min<
            typename pmacc::math::CT::mul<
                SuperCellSize,
                GuardSize
            >::type
        >::type::value >= currentUpperMargin
    );

    float_X charge;

    /* At the moment Esirkepov only support YeeCell were W is defined at origin (0,0,0)
     *
     * \todo: please fix me that we can use CenteredCell
     */
    template<
        typename DataBoxJ,
        typename PosType,
        typename VelType,
        typename ChargeType,
        typename T_Acc,
        typename T_MatA,
        typename T_MatB
    >
    DINLINE void operator()(
        T_Acc const & acc,
        DataBoxJ dataBoxJ,
        const PosType pos,
        const VelType velocity,
        const ChargeType charge,
        const float_X deltaTime,
        T_MatA & matA,
        T_MatB & matB,
        int tid,
        const bool parIsValid
    )
    {
        this->charge = charge;

        const float3_X deltaPos = float3_X(velocity.x() * deltaTime / cellSize.x(),
                                           velocity.y() * deltaTime / cellSize.y(),
                                           velocity.z() * deltaTime / cellSize.z());
        const PosType oldPos = pos - deltaPos;
        Line<float3_X> line(oldPos, pos);

        DataSpace<DIM3> gridShift;
        if( parIsValid )
        {
            /* calculate the offset for the virtual coordinate system */
            for(int d=0; d<simDim; ++d)
            {
                int iStart;
                int iEnd;
                constexpr bool isSupportEven = ( supp % 2 == 0 );
                RelayPoint< isSupportEven >()(
                    iStart,
                    iEnd,
                    line.m_pos0[d],
                    line.m_pos1[d]
                );
                gridShift[d] = iStart < iEnd ? iStart : iEnd; // integer min function
                /* shift the particle position to the virtual coordinate system */
                line.m_pos0[d] -= gridShift[d];
                line.m_pos1[d] -= gridShift[d];
            }
        }
        /* shift current field to the virtual coordinate system */
        auto cursorJ = dataBoxJ.shift(gridShift);

        cptCurrent(
            acc,
            cursorJ,
            line,
            matA,
            matB,
            tid,
            parIsValid
        );
    }

    template< typename MatA, typename MatB, typename MatResult>
    DINLINE void matmul(MatA & matA, MatB & matB, MatResult & matResult, int tid)
    {
#if (PIC_USE_MMA == 1)
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;
        nvcuda::wmma::fill_fragment(acc_frag, float(0.0));
        nvcuda::wmma::load_matrix_sync(a_frag, matA.getPointer(), 16);
        nvcuda::wmma::load_matrix_sync(b_frag, matB.getPointer(), 16);
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        nvcuda::wmma::store_matrix_sync(matResult.getPointer() , acc_frag, 16, nvcuda::wmma::mem_row_major);
#else
        PMACC_CASSERT_MSG(
            __PICONGPU_NUMWARPS_must_be_one_if_non_mma_are_used,
            PICONGPU_NUMWARPS == 1
        );
        __syncwarp();
        constexpr int matSize = 16;

        int col = tid % matSize;
        int row = tid / matSize;

        // 32 == number of threads working on the matrix
        for( int k = row; k < matSize; k += 32 / matSize)
        {
            float_X r = 0.0;
            for( int i = 0; i < matSize; i++ )
            {
                r += __half2float(__hmul(matA( DataSpace< DIM2>( k, i ) ),  matB( DataSpace< DIM2>( col, i ) )));
            }
            matResult( DataSpace< DIM2>( col, k ) ) = r;
        }
        __syncwarp();
#endif
    }

#if (PIC_ENFORCE_SHARED_ATOMICS == 1)
    DINLINE void atomicAddSharedMem( float* addr, float const & val )
    {
        asm volatile("{\n.reg .u64 rd1;"
                     ".reg .u32 t1;.reg .f32 t2;"
                     "cvta.to.shared.u64 rd1,%0;" :: "l"(addr));
        asm volatile("cvt.u32.u64 t1,rd1;"
                     "atom.shared.add.f32 t2, [t1], %0;\n}" :: "f"(val));

    }
#endif

    /**
     * deposites current in z-direction
     *
     * \param cursorJ cursor pointing at the current density field of the particle's cell
     * \param line trajectory of the particle from to last to the current time step
     * \param cellEdgeLength length of edge of the cell in z-direction
     */
    template<
        typename CursorJ,
        typename T_Acc,
        typename T_MatA,
        typename T_MatB
    >
    DINLINE void cptCurrent(
        T_Acc const & acc,
        CursorJ cursorJ,
        Line<float3_X>& line,
        T_MatA & matA,
        T_MatB & matB,
        int tid,
        bool parIsValid
    )
    {
        /* skip calculation if the particle is not moving */
        if(parIsValid && line.m_pos0 == line.m_pos1)
            parIsValid = false;

        constexpr int begin = -currentLowerMargin + 1;
        constexpr int end = begin + supp;

        /* pick every cell in the xy-plane that is overlapped by particle's
         * form factor and deposit the current for the cells above and beneath
         * that cell and for the cell itself.
         *
         * for loop optimization (help the compiler to generate better code):
         *   - use a loop with a static range
         *   - skip invalid indexes with a if condition around the full loop body
         *     ( this helps the compiler to mask threads without work )
         */

        // split the warp
        int const idx = tid % 16;
        /* the second batch in the warp is working on a 8x8 matrix within the 16x16
         * matrix with an offset 8,8
         */
        int const batch = tid / 16;
        int const matRow = idx % 4;
        int const i = begin + matRow;

        __syncwarp();

        Set< float > setMatZero(
            0.0
        );

        // define size of result matrix
        using MatResult = SuperCellDescription<
            pmacc::math::CT::Int<
                16,
                16
            >
        >;
        ThreadCollective<
            MatResult,
            32 // one warp
        > collectiveSetMatZero( tid );

        DataBox<
            SharedBox<
                float,
                pmacc::math::CT::Int<
                   16,
                   16
                >
            >
        > matResult(
            SharedBox<
                float,
                pmacc::math::CT::Int<
                   16,
                   16
                >
            >{ (float*)matA.getPointer() }
        );

        // because of the reuse of the matrix A and B as result matrix it must be cleared each round
        collectiveSetMatZero( acc, setMatZero, matResult );


        // wait that previous round is flushed to the charge current matrix in the shared memory
        __syncwarp();


        if( parIsValid && idx < 12 )
        {
            /* offset: S1  = (0,0) -> threads 0-3
             *         S2  = (4,0) -> threads 4-7
             *         S1  = (4,4) -> threads 8-12
             */
            DataSpace< DIM2 > const offset = DataSpace< DIM2 >(
                ( idx < 4 ? 0 : 4 ) + batch * 8,
                ( idx < 8 ? 0 : 4 ) + batch * 8
            );

            /* direction: S1  = 0 -> threads 0-3
             *            S2  = 1 -> threads 4-7
             *            S1  = 0 -> threads 8-11
             */
            uint32_t const direction = ( idx / 4 ) % 2;

            float_X const s0i = S0( line, i, direction );
            float_X const dsi = S1( line, i, direction ) - s0i;

            // column, row
            matA( DataSpace< DIM2 >( matRow, 0 ) + offset ) = __float2half(s0i);
            matA( DataSpace< DIM2 >( matRow, 1 ) + offset ) = __float2half(0.5_X * dsi);
            matA( DataSpace< DIM2 >( matRow, 2 ) + offset ) = __float2half(0.5_X * s0i);
            matA( DataSpace< DIM2 >( matRow, 3 ) + offset ) = __float2half(1.0_X / 3.0_X * dsi);
        }

        if( parIsValid && idx < 8 )
        {
            /* offset: S3  = (0,0) -> threads 0-3
             *         S2  = (4,4) -> threads 4-7
             */
            DataSpace< DIM2 > const offset = DataSpace< DIM2 >::create(
                ( idx < 4 ? 0 : 4 ) + batch * 8
            );

            /* direction: S3  = 2 -> threads 0-3
             *            S2  = 1 -> threads 4-7
             */
            uint32_t const direction = idx  < 4 ? 2 : 1;

            float_X const s0j = S0( line, i, direction );
            float_X const dsj = S1( line, i, direction ) - s0j;

            // column, row
            matB( DataSpace< DIM2 >( matRow, 0 ) + offset ) = __float2half(s0j);
            matB( DataSpace< DIM2 >( matRow, 1 ) + offset ) = __float2half(s0j);
            matB( DataSpace< DIM2 >( matRow, 2 ) + offset ) = __float2half(dsj);
            matB( DataSpace< DIM2 >( matRow, 3 ) + offset ) = __float2half(dsj);
        }

        // wait that matrix is filled
        __syncwarp();

        // this function is also syncing the warp
        matmul( matA, matB, matResult, tid );

        // idx is between [0;16)
        DataSpace< DIM2 > id2d( idx % 4 , idx / 4 );

        #pragma unroll 2
        for( int b = 0; b < 2; ++b )
        {
            bool const processParticle = ::pmacc::nvidia::warpBroadcast( parIsValid, b * 16 );

            if( processParticle )
            {

                const float_X c = ::pmacc::nvidia::warpBroadcast( this->charge, b * 16 );
                /* We multiply with `cellEdgeLength` due to the fact that the attribute for the
                 * in-cell particle `position` (and it's change in DELTA_T) is normalize to [0,1)
                 */
                const float3_X currentSurfaceDensity = c * (float_X(1.0) / float_X(CELL_VOLUME * DELTA_T)) * cellSize;

                PMACC_CASSERT_MSG_TYPE(
                    cursor_size_is_not_a_multiple_of_4_byte,
                    bmpl::int_< sizeof( CursorJ ) > ,
                    sizeof( CursorJ ) % 4 == 0
                );

                /* Provide all threads with the cursor (iterator) which is already
                 * shifted to the correct J field offset for the particle handled by the current batch.
                 */
                CursorJ batchCursorJ(cursorJ);
                for( int i = 0; i < sizeof(CursorJ) / 4; ++i )
                {
                    uint32_t const value = *( reinterpret_cast< uint32_t* >( &cursorJ ) + i );
                    *( reinterpret_cast< uint32_t* >(&batchCursorJ) + i ) = ::pmacc::nvidia::warpBroadcast( value, b * 16 );
                }

                PMACC_CASSERT_MSG_TYPE(
                    Line_size_is_not_a_multiple_of_4_byte,
                    bmpl::int_< sizeof( Line< float3_X > ) > ,
                    sizeof( Line< float3_X > ) % 4 == 0
                );

                // temporary storage of the trajectory of the particle from the current batch
                Line< float3_X > parLine;
                for( int i = 0; i < sizeof( Line< float3_X > ) / 4; ++i )
                {
                    uint32_t const value = *( reinterpret_cast< uint32_t* >( &line ) + i );
                    *( reinterpret_cast< uint32_t* >( &parLine ) + i ) = ::pmacc::nvidia::warpBroadcast( value, b * 16 );
                }

                if( tid < 32 )
                {
                    /* 1 -> thread 0-15
                     * 2 -> thread 16-31
                     */
                    int w = tid < 16 ? 1 : 2;
                    /* offset: W2  = (0,0) -> threads 0-15
                     *         W3  = (4,4) -> threads 16-31
                     */
                    DataSpace< DIM2 > const offset = DataSpace< DIM2 >(
                        ( w == 1 ? 0 : 4 ) + b * 8 ,
                        ( w == 1 ? 0 : 4 ) + b * 8
                    );
                    float_X accumulated_J( 0.0 );

                    float_X const tmp = -currentSurfaceDensity[ w ] * ( matResult( id2d + offset ) );
                    /* we cheat a little bit y and z can be set both to the id2d.x
                     * because the correct direction will be later overwritten with k
                     */
                    DataSpace< DIM3 > jOffset(begin + id2d.y(), begin + id2d.x(), begin + id2d.x() );
                    for( int k = begin ; k < end; ++k )
                    {
                        /* This is the implementation of the FORTRAN W(i,j,k,3)/ C style W(i,j,k,2) version from
                         * Esirkepov paper. All coordinates are rotated before thus we can
                         * always use C style W(i,j,k,2).
                         */
                        jOffset[ w ] = k;
                        accumulated_J += DS( parLine, k, w ) * tmp;
#if (PIC_ENFORCE_SHARED_ATOMICS == 1)
                        atomicAddSharedMem(&(batchCursorJ( jOffset )[ w ]), accumulated_J);
#else
                        atomicAdd(&(batchCursorJ( jOffset )[ w ]), accumulated_J, ::alpaka::hierarchy::Threads{});
#endif
                    }
                }

                if( tid < 16 )
                {
                    float_X accumulated_J( 0.0 );
                    DataSpace< DIM2 > matOffset = DataSpace< DIM2 >( b * 8 , 4 + b * 8 );
                    float_X const tmp = -currentSurfaceDensity.x() * ( matResult( id2d + matOffset ) );
                    DataSpace< DIM3 > jOffset( 0, begin + id2d.y(), begin + id2d.x() );
                    for( int k = begin ; k < end; ++k )
                    {
                        /* This is the implementation of the FORTRAN W(i,j,k,3)/ C style W(i,j,k,2) version from
                         * Esirkepov paper. All coordinates are rotated before thus we can
                         * always use C style W(i,j,k,2).
                         */
                        accumulated_J += DS( parLine, k, 0 ) * tmp;
                        jOffset.x() = k;
#if (PIC_ENFORCE_SHARED_ATOMICS == 1)
                        atomicAddSharedMem(&(batchCursorJ( jOffset ).x()), accumulated_J);
#else
                        atomicAdd(&(batchCursorJ( jOffset ).x()), accumulated_J, ::alpaka::hierarchy::Threads{});
#endif
                    }
                }
            }
        }
    }

    /** calculate S0 (see paper)
     * @param line element with previous and current position of the particle
     * @param gridPoint used grid point to evaluate assignment shape
     * @param d dimension range {0,1,2} means {x,y,z}
     *          different to Esirkepov paper, here we use C style
     */
    DINLINE float_X S0(const Line<float3_X>& line, const float_X gridPoint, const uint32_t d)
    {
        return ParticleAssign()(gridPoint - line.m_pos0[d]);
    }

   /** calculate S1 (see paper)
     * @param line element with previous and current position of the particle
     * @param gridPoint used grid point to evaluate assignment shape
     * @param d dimension range {0,1,2} means {x,y,z}
     *          different to Esirkepov paper, here we use C style
     */
    DINLINE float_X S1(const Line<float3_X>& line, const float_X gridPoint, const uint32_t d)
    {
        return ParticleAssign()(gridPoint - line.m_pos1[d]);
    }

    /** calculate DS (see paper)
     * @param line element with previous and current position of the particle
     * @param gridPoint used grid point to evaluate assignment shape
     * @param d dimension range {0,1,2} means {x,y,z}]
     *          different to Esirkepov paper, here we use C style
     */
    DINLINE float_X DS(const Line<float3_X>& line, const float_X gridPoint, const uint32_t d)
    {
        return ParticleAssign()(gridPoint - line.m_pos1[d]) - ParticleAssign()(gridPoint - line.m_pos0[d]);
    }

    static pmacc::traits::StringProperty getStringProperties()
    {
        pmacc::traits::StringProperty propList( "name", "Esirkepov" );
        return propList;
    }
};

} //namespace currentSolver

} //namespace picongpu

#include "picongpu/fields/currentDeposition/Esirkepov/Esirkepov2D.hpp"

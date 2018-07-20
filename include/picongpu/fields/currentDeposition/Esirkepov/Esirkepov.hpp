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
#include <pmacc/types.hpp>
#include <pmacc/cuSTL/cursor/Cursor.hpp>
#include <pmacc/cuSTL/cursor/tools/twistVectorFieldAxes.hpp>
#include <pmacc/cuSTL/cursor/compile-time/SafeCursor.hpp>
#include <pmacc/nvidia/atomic.hpp>

#include "picongpu/fields/currentDeposition/Esirkepov/Esirkepov.def"
#include "picongpu/fields/currentDeposition/Esirkepov/Line.hpp"
#include "picongpu/fields/currentDeposition/RelayPoint.hpp"


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
        typename T_MatB,
        typename T_MatR
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
        T_MatR & matResult,
        int idx
    )
    {
        this->charge = charge;
        const float3_X deltaPos = float3_X(velocity.x() * deltaTime / cellSize.x(),
                                           velocity.y() * deltaTime / cellSize.y(),
                                           velocity.z() * deltaTime / cellSize.z());
        const PosType oldPos = pos - deltaPos;
        Line<float3_X> line(oldPos, pos);

        DataSpace<DIM3> gridShift;

        /* Define in which direction the particle leaves the cell.
         * It is not relevant whether the particle leaves the cell via
         * the positive or negative cell border.
         *
         * 0 == stay in cell
         * 1 == leave cell
         */
        DataSpace<simDim> leaveCell;

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
            /* particle is leaving the cell */
            leaveCell[d] = iStart != iEnd ? 1 : 0;
            /* shift the particle position to the virtual coordinate system */
            line.m_pos0[d] -= gridShift[d];
            line.m_pos1[d] -= gridShift[d];
        }
        /* shift current field to the virtual coordinate system */
        auto cursorJ = dataBoxJ.shift(gridShift).toCursor();


        cptCurrent(
            acc,
            leaveCell,
            cursorJ,
            line,
            matA,
            matB,
            matResult,
            idx
        );
    }

    template< typename MatA, typename MatB, typename MatResult>
    HDINLINE void matmul(MatA const & matA, MatB const & matB, MatResult & matResult, int tid)
    {
        constexpr int matSize = 8;

        int col = tid % matSize;
        int row = tid / matSize;


        for( int k = row; k < matSize; k+=4)
        {
            float_X r = 0.0;
            for( int i = 0; i < matSize; i++ )
            {
                r += matA( DataSpace< DIM2>( k, i ) ) *  matB( DataSpace< DIM2>( col, i ) );
            }
            matResult( DataSpace< DIM2>( col, k ) ) = r;
        }

    }

    template< typename Mat>
    HDINLINE void pMat(Mat const & matA)
    {
        constexpr int matSize = 8;


        for( int k = 0; k < matSize; k++)
        {
            for( int i = 0; i < matSize; i++ )
            {
                printf("%f ",matA( DataSpace< DIM2>( i, k ) ));
            }
            printf("\n");
        }
        printf("-----\n");

    }

    /**
     * deposites current in z-direction
     *
     * \param leaveCell vector with information if the particle is leaving the cell
     *         (for each direction, 0 means stays in cell and 1 means leaves cell)
     * \param cursorJ cursor pointing at the current density field of the particle's cell
     * \param line trajectory of the particle from to last to the current time step
     * \param cellEdgeLength length of edge of the cell in z-direction
     */
    template<
        typename CursorJ,
        typename T_Acc,
        typename T_MatA,
        typename T_MatB,
        typename T_MatR
    >
    DINLINE void cptCurrent(
        T_Acc const & acc,
        const DataSpace<simDim>& leaveCell,
        CursorJ cursorJ,
        const Line<float3_X>& line,
        T_MatA & matA,
        T_MatB & matB,
        T_MatR & matResult,
        int idx
    )
    {
        /* skip calculation if the particle is not moving */
        if(line.m_pos0 == line.m_pos1)
            return;

        constexpr int begin = -currentLowerMargin + 1;
        constexpr int end = begin + supp;

        /* We multiply with `cellEdgeLength` due to the fact that the attribute for the
         * in-cell particle `position` (and it's change in DELTA_T) is normalize to [0,1)
         */
        const float3_X currentSurfaceDensity = this->charge * (float_X(1.0) / float_X(CELL_VOLUME * DELTA_T)) * cellSize;

        /* pick every cell in the xy-plane that is overlapped by particle's
         * form factor and deposit the current for the cells above and beneath
         * that cell and for the cell itself.
         *
         * for loop optimization (help the compiler to generate better code):
         *   - use a loop with a static range
         *   - skip invalid indexes with a if condition around the full loop body
         *     ( this helps the compiler to mask threads without work )
         */


        if( begin + idx < end  + 1 )
        {
            DataSpace< DIM2 > matOffset = DataSpace< DIM2 >( 0, 0 );
            DataSpace< DIM2 > matOffset2 = DataSpace< DIM2 >( 4, 4 );
            int i = begin + idx;

            const float_X s0i = S0( line, i, 0 );
            const float_X dsi = S1( line, i, 0 ) - s0i;

            // column, row
            matA( DataSpace< DIM2 >( idx, 0 ) + matOffset ) = s0i;
            matA( DataSpace< DIM2 >( idx, 0 ) + matOffset2 ) = s0i;
            matA( DataSpace< DIM2 >( idx, 1 ) + matOffset ) = 0.5_X * dsi;
            matA( DataSpace< DIM2 >( idx, 1 ) + matOffset2 ) = 0.5_X * dsi;
            matA( DataSpace< DIM2 >( idx, 2 ) + matOffset ) = 0.5_X * s0i;
            matA( DataSpace< DIM2 >( idx, 2 ) + matOffset2 ) = 0.5_X * s0i;
            matA( DataSpace< DIM2 >( idx, 3 ) + matOffset ) = 1.0_X / 3.0_X * dsi;
            matA( DataSpace< DIM2 >( idx, 3 ) + matOffset2 ) = 1.0_X / 3.0_X * dsi;
        }

        if( begin + idx < end  + 1 )
        {
            DataSpace< DIM2 > matOffsetA = DataSpace< DIM2 >( 4, 0 );
            DataSpace< DIM2 > matOffsetB = DataSpace< DIM2 >( 4, 4 );
            int j = begin + idx;

            const float_X s0j = S0( line, j, 1 );
            const float_X dsj = S1( line, j, 1 ) - s0j;

            // column, row
            matA( DataSpace< DIM2 >( idx, 0 ) + matOffsetA ) = s0j;
            matB( DataSpace< DIM2 >( idx, 0 ) + matOffsetB ) = s0j;
            matA( DataSpace< DIM2 >( idx, 1 ) + matOffsetA ) = 0.5_X * dsj;
            matB( DataSpace< DIM2 >( idx, 1 ) + matOffsetB ) = s0j;
            matA( DataSpace< DIM2 >( idx, 2 ) + matOffsetA ) = 0.5_X * s0j;
            matB( DataSpace< DIM2 >( idx, 2 ) + matOffsetB ) = dsj;
            matA( DataSpace< DIM2 >( idx, 3 ) + matOffsetA ) = 1.0_X / 3.0_X * dsj;
            matB( DataSpace< DIM2 >( idx, 3 ) + matOffsetB ) = dsj;
        }

        if( begin + idx < end  + 1 )
        {
            DataSpace< DIM2 > matOffset = DataSpace< DIM2 >( 0, 0 );
            int k = begin + idx;

            const float_X s0k = S0( line, k, 2 );
            const float_X dsk = S1( line, k, 2 ) - s0k;

            // column, row
            matB( DataSpace< DIM2 >( idx, 0 ) + matOffset ) = s0k;
            matB( DataSpace< DIM2 >( idx, 1 ) + matOffset ) = s0k;
            matB( DataSpace< DIM2 >( idx, 2 ) + matOffset ) = dsk;
            matB( DataSpace< DIM2 >( idx, 3 ) + matOffset ) = dsk;
        }

        __syncthreads();

        matmul( matA, matB, matResult, idx );
#if 0
        if(blockIdx.x == 0 && blockIdx.y == 0 &&blockIdx.z == 0 &&threadIdx.x == 0 &&threadIdx.y == 0 &&threadIdx.z == 0)
        {
            printf("A:\n");
            pMat(matA);
            printf("B:\n");
            pMat(matB);
        }

        matmul( matA, matB, matResult, idx );

        __syncthreads();

        if(blockIdx.x == 0 && blockIdx.y == 0 &&blockIdx.z == 0 &&threadIdx.x == 0 &&threadIdx.y == 0 &&threadIdx.z == 0)
        {
            printf("C:\n");
            pMat(matResult);
        }
#endif
        DataSpace< DIM2 > id2d( idx % 4 , idx / 4 );

        if( idx < 16 )
        {
            float_X accumulated_J( 0.0 );
            DataSpace< DIM2 > matOffset = DataSpace< DIM2 >( 0, 4 );
            float_X const tmp = -currentSurfaceDensity.x() * matResult( id2d + matOffset );
            for( int k = begin ; k < end; ++k )
            {
                /* This is the implementation of the FORTRAN W(i,j,k,3)/ C style W(i,j,k,2) version from
                 * Esirkepov paper. All coordinates are rotated before thus we can
                 * always use C style W(i,j,k,2).
                 */
                accumulated_J += DS( line, k, 0 ) * tmp;
                atomicAdd( &( ( *cursorJ( k, begin + id2d.y(), begin + id2d.x() ) ).x() ), accumulated_J, ::alpaka::hierarchy::Threads{} );
            }
        }

        if( idx < 16 )
        {
            float_X accumulated_J( 0.0 );
            DataSpace< DIM2 > matOffset = DataSpace< DIM2 >( 0, 0 );
            float_X const tmp = -currentSurfaceDensity.y() * matResult( id2d + matOffset );
            for( int k = begin ; k < end; ++k )
            {
                /* This is the implementation of the FORTRAN W(i,j,k,3)/ C style W(i,j,k,2) version from
                 * Esirkepov paper. All coordinates are rotated before thus we can
                 * always use C style W(i,j,k,2).
                 */
                accumulated_J += DS( line, k, 1 ) * tmp;
                atomicAdd( &( ( *cursorJ( begin + id2d.y(), k, begin + id2d.x() ) ).y() ), accumulated_J, ::alpaka::hierarchy::Threads{} );
            }
        }

        if( idx < 16 )
        {
            float_X accumulated_J( 0.0 );
            DataSpace< DIM2 > matOffset = DataSpace< DIM2 >( 4, 4 );
            float_X const tmp = -currentSurfaceDensity.z() * matResult( id2d + matOffset );
            for( int k = begin ; k < end; ++k )
            {
                /* This is the implementation of the FORTRAN W(i,j,k,3)/ C style W(i,j,k,2) version from
                 * Esirkepov paper. All coordinates are rotated before thus we can
                 * always use C style W(i,j,k,2).
                 */
                accumulated_J += DS( line, k, 2 ) * tmp;
                atomicAdd( &( ( *cursorJ( begin + id2d.y(), begin + id2d.x(), k ) ).z() ), accumulated_J, ::alpaka::hierarchy::Threads{} );
            }
        }
        __syncthreads();
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

/* Copyright 2014-2019 Rene Widera
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

#include <pmacc/mappings/threads/ForEachIdx.hpp>
#include <pmacc/mappings/threads/IdxConfig.hpp>
#include <pmacc/mappings/threads/WorkerCfg.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/nvidia/rng/RNG.hpp>
#include <pmacc/random/RNGProvider.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

#include "picongpu/particles/filter/filter.def"
#include "picongpu/particles/collision/IBinary.def"
#include "picongpu/particles/collision/binary/MomentumSwap.hpp"



namespace picongpu
{
namespace particles
{
namespace collision
{
namespace detail
{
    struct ListEntry
    {
        uint32_t size;
        uint32_t* ptrToIndicies;

        template< typename T_DeviceHeapHandle >
        DINLINE void init( T_DeviceHeapHandle & deviceHeapHandle, uint32_t numPar )
        {
            ptrToIndicies = nullptr;
            if( numPar != 0u )
            {
                // printf("alloc %u: %u\n", linearIdx, (nppc[ linearIdx ] + 1) );
#if( PMACC_CUDA_ENABLED == 1 )
                int i = 0;
                while( ptrToIndicies == nullptr )
                {
                    ptrToIndicies = (uint32_t*) deviceHeapHandle.malloc( sizeof(uint32_t) * numPar );
                    if(i >=5)
                        printf("no memory: %u\n",numPar);
                    ++i;
                }
#else
                ptrToIndicies = new uint32_t[ numPar ];
#endif
            }
            //reset counter
            size = 0u;
        }

        template< typename T_DeviceHeapHandle >
        DINLINE void finalize( T_DeviceHeapHandle & deviceHeapHandle )
        {
            if(ptrToIndicies != nullptr)
            {
#if( PMACC_CUDA_ENABLED == 1 )
                deviceHeapHandle.free( (void*) ptrToIndicies );
                ptrToIndicies = nullptr;
#else
                delete( ptrToIndicies );
#endif
            }
        }

        // non collective
        template< typename T_Acc, typename T_RngHandle >
        DINLINE void shuffle( T_Acc const & acc, T_RngHandle & rngHandle)
        {
            using UniformUint32_t = pmacc::random::distributions::Uniform<uint32_t>;
            auto rng = rngHandle.template applyDistribution< UniformUint32_t >();
            // shuffle the particle lookup table
            for(uint32_t i = size; i > 1; --i)
            {
                /* modulo is not perfect but okish,
                 * because of the loop head mod zero is not possible
                 */
                int p = rng(acc) % i;
                if( i - 1 != p )
                    swap(ptrToIndicies[ i - 1 ], ptrToIndicies[ p ]);
            }
        }

    private:
        DINLINE void swap( uint32_t & v0, uint32_t & v1)
        {
            uint32_t tmp = v0;
            v0 = v1;
            v1 = tmp;
        }
    };

    template< typename T_Acc, typename T_ForEach, typename T_ParBox, typename T_FramePtr, typename T_Array>
    DINLINE void particlesCntHistogram(
        T_Acc const & acc,
        T_ForEach forEach,
        T_ParBox & parBox,
        T_FramePtr frame,
        uint32_t const numParticlesInSupercell,
        T_Array& nppc
    )
    {
            using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
            constexpr uint32_t frameSize = pmacc::math::CT::volume< SuperCellSize >::type::value;

            for(uint32_t i = 0; i < numParticlesInSupercell; i += frameSize)
            {

                forEach(
                    [&](
                        uint32_t const linearIdx,
                        uint32_t const idx
                    )
                    {
                        if( i + linearIdx < numParticlesInSupercell)
                        {
                            auto particle = frame[ linearIdx ];
                            auto parLocalIndex = particle[ localCellIdx_ ];
                            if(parLocalIndex < 256)
                            {

                                atomicAdd( &nppc[ parLocalIndex ], 1u);
                            }
#if 1
                            else
                                printf("nooooo %u\n",parLocalIndex);
#endif
                        }
                    }
                );
                frame = parBox.getNextFrame( frame );
            }
    }

    template< typename T_Acc, typename T_ForEach, typename T_ParBox, typename T_FramePtr, typename T_EntryListArray>
    DINLINE void updateLinkedList(
        T_Acc const & acc,
        T_ForEach forEach,
        T_ParBox & parBox,
        T_FramePtr frame,
        uint32_t const numParticlesInSupercell,
        T_EntryListArray & parCellList
    )
    {
        using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
        constexpr uint32_t frameSize = pmacc::math::CT::volume< SuperCellSize >::type::value;
        for(uint32_t i = 0; i < numParticlesInSupercell; i += frameSize)
        {

            forEach(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const idx
                )
                {
                    uint32_t const parInSuperCellIdx = i + linearIdx;
                    if( parInSuperCellIdx < numParticlesInSupercell )
                    {
                        auto particle = frame[ linearIdx ];
                        auto parLocalIndex = particle[ localCellIdx_ ];
                        uint32_t parOffset = atomicAdd( &parCellList[ parLocalIndex ].size , 1u );
                        parCellList[ parLocalIndex ].ptrToIndicies[ parOffset ] = parInSuperCellIdx;
                    }
                }
            );
            frame = parBox.getNextFrame( frame );
        }
    }

    template<
        typename T_Acc,
        typename T_RngHandle
    >
    struct  CollisionContext
    {
        T_Acc const * m_acc;
        mutable T_RngHandle* m_hRng;

        DINLINE CollisionContext(T_Acc const & acc, T_RngHandle & hRng) : m_acc(&acc), m_hRng(&hRng)
        {

        }

    };

    template<
        typename T_Acc,
        typename T_RngHandle
    >
    DINLINE CollisionContext< T_Acc, T_RngHandle> makeCollisionContext(T_Acc const & acc, T_RngHandle & hRng)
    {
        return CollisionContext< T_Acc, T_RngHandle>(acc, hRng);
    }

} // namespace detail

    template< uint32_t T_numWorkers >
    struct Collision
    {
        template<
            typename T_ParBox,
            typename T_Mapping,
            typename T_Acc,
            typename T_DeviceHeapHandle,
            typename T_RngHandle,
            typename T_CollisionFunctor
        >
        DINLINE void operator()(
            T_Acc const & acc,
            T_ParBox pb,
            T_Mapping const mapper,
            T_DeviceHeapHandle deviceHeapHandle,
            T_RngHandle rngHandle,
            T_CollisionFunctor const collisionFunctor
        ) const
        {
            using namespace pmacc::particles::operations;
            using namespace mappings::threads;

            using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
            constexpr uint32_t frameSize = pmacc::math::CT::volume< SuperCellSize >::type::value;
            constexpr uint32_t numWorkers = T_numWorkers;

            using FramePtr = typename T_ParBox::FramePtr;

            PMACC_SMEM(
                acc,
                nppc,
                memory::Array<
                    uint32_t,
                    frameSize
                >
            );

            PMACC_SMEM(
                acc,
                parCellList,
                memory::Array<
                    detail::ListEntry,
                    frameSize
                >
            );

            uint32_t const workerIdx = threadIdx.x;

            using FrameDomCfg = IdxConfig<
                frameSize,
                numWorkers
            >;

            DataSpace< simDim > const superCellIdx = mapper.getSuperCellIndex( DataSpace< simDim > ( blockIdx ) );

            // offset of the superCell (in cells, without any guards) to the origin of the local domain
            DataSpace< simDim > const localSuperCellOffset =
            superCellIdx - mapper.getGuardingSuperCells( );
            rngHandle.init(
                localSuperCellOffset * SuperCellSize::toRT() +
                DataSpaceOperations< simDim >::template map< SuperCellSize >( workerIdx )
            );

            auto & superCell = pb.getSuperCell( superCellIdx );
            uint32_t numParticlesInSupercell = superCell.getNumParticles();


            /* loop over all particles in the frame */
            ForEachIdx< FrameDomCfg > forEachFrameElem( workerIdx );

            forEachFrameElem(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const
                )
                {
                    nppc[ linearIdx ] = 0u;
                }
            );

            __syncthreads();

            FramePtr firstFrame = pb.getFirstFrame( superCellIdx );
            detail::particlesCntHistogram( acc, forEachFrameElem, pb, firstFrame, numParticlesInSupercell, nppc );

            __syncthreads();

            // memory for particle indices
            forEachFrameElem(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const
                )
                {
                    parCellList[ linearIdx ].init( deviceHeapHandle,  nppc[ linearIdx ] );
                }
            );

            __syncthreads();

            detail::updateLinkedList( acc, forEachFrameElem, pb, firstFrame, numParticlesInSupercell, parCellList );

            __syncthreads();
#if 1
            if(threadIdx.x == 0)
            {
                for(int i=0;i<256;++i)
                    if( nppc[ i ] != parCellList[ i ].size)
                        printf("ppc %i: %u == %u\n",i, nppc[ i ], parCellList[ i ].size);
            }
            __syncthreads();
#endif

            //shuffle  indices list
            forEachFrameElem(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const idx
                )
                {
                    parCellList[ linearIdx ].shuffle( acc, rngHandle );
                }
            );

            auto accFunctor = collisionFunctor(
                acc,
                localSuperCellOffset,
                WorkerCfg< T_numWorkers >{ workerIdx }
            );

            forEachFrameElem(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const idx
                )
                {
                    uint32_t const numParPerCell = parCellList[ linearIdx ].size;

                    // skip particle offset counter
                    uint32_t* parListStart = parCellList[ linearIdx ].ptrToIndicies;

                    if(numParPerCell != 0)
                        for(uint32_t i = 0; i < numParPerCell - 1u; i += 2)
                        {
#if 1
                            if(parListStart[ i ] >= numParticlesInSupercell || parListStart[ i + 1 ] >= numParticlesInSupercell)
                                printf("wrong %u %u of %u\n", parListStart[ i ],parListStart[ i  + 1],numParPerCell );
#endif
                            auto srcPar = getParticle(pb, firstFrame, parListStart[ i ]);
                            auto destPar = getParticle(pb, firstFrame, parListStart[ i + 1 ]);
                            accFunctor(detail::makeCollisionContext(acc,rngHandle), srcPar, destPar);
                        }
                }
            );

            __syncthreads();

            forEachFrameElem(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const
                )
                {
                    parCellList[ linearIdx ].finalize( deviceHeapHandle );
                }
            );

        }

        template< typename T_ParBox, typename T_FramePtr >
        DINLINE auto getParticle(T_ParBox& parBox, T_FramePtr frame, uint32_t particleId) const -> typename T_FramePtr::type::ParticleType
        {
            constexpr uint32_t frameSize = pmacc::math::CT::volume< typename T_FramePtr::type::SuperCellSize >::type::value;
            uint32_t const skipFrames = particleId / frameSize;
            for(uint32_t i = 0; i < skipFrames; ++i)
                frame = parBox.getNextFrame( frame );
            return frame[ particleId % frameSize ];
        }
    };

    template<
        typename T_CollisionFunctor,
        typename T_Species,
        typename T_Filter = filter::All
    >
    struct DoCollision
    {
        void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap, uint32_t currentStep)
        {
             using Species = pmacc::particles::meta::FindByNameOrType_t<
                VectorAllSpecies,
                T_Species
            >;
            using FrameType = typename Species::FrameType;

            using CollisionFunctor = typename bmpl::apply1<
                T_CollisionFunctor,
                Species
            >::type;
            using FilteredCollisionFunctor = IBinary<
                CollisionFunctor,
                T_Filter
            >;

            DataConnector &dc = Environment<>::get().DataConnector();
            auto species = dc.get< Species >( FrameType::getName(), true );

            AreaMapping<
                CORE + BORDER,
                picongpu::MappingDesc
            > mapper( species->getCellDescription() );

            constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                pmacc::math::CT::volume< SuperCellSize >::type::value
            >::value;

            /* random number generator */
            using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;

            PMACC_KERNEL( Collision< numWorkers >{ } )(
                mapper.getGridDim(),
                numWorkers
            )(
                species->getDeviceParticlesBox( ),
                mapper,
                deviceHeap->getAllocatorHandle(),
                RNGFactory::createHandle(),
                FilteredCollisionFunctor(currentStep)
            );

            dc.releaseData( FrameType::getName() );
        }
    };

} // namespace collision
} // namespace particles
} // namespace picongpu

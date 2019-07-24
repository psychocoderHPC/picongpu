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
#include "picongpu/particles/collision/IntraSpecies.hpp"



namespace picongpu
{
namespace particles
{
namespace collision
{
    template< uint32_t T_numWorkers >
    struct InterCollision
    {
        template<
            typename T_ParBox0,
            typename T_ParBox1,
            typename T_Mapping,
            typename T_Acc,
            typename T_DeviceHeapHandle,
            typename T_RngHandle,
            typename T_CollisionFunctor
        >
        DINLINE void operator()(
            T_Acc const & acc,
            T_ParBox0 pb0,
            T_ParBox1 pb1,
            T_Mapping const mapper,
            T_DeviceHeapHandle deviceHeapHandle,
            T_RngHandle rngHandle,
            T_CollisionFunctor const collisionFunctor
        ) const
        {
            using namespace pmacc::particles::operations;
            using namespace mappings::threads;

            using SuperCellSize = typename T_ParBox0::FrameType::SuperCellSize;
            constexpr uint32_t frameSize = pmacc::math::CT::volume< SuperCellSize >::type::value;
            constexpr uint32_t numWorkers = T_numWorkers;

            using FramePtr0 = typename T_ParBox0::FramePtr;
            using FramePtr1 = typename T_ParBox1::FramePtr;

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
                parCellList0,
                memory::Array<
                    detail::ListEntry,
                    frameSize
                >
            );
            PMACC_SMEM(
                acc,
                parCellList1,
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

            auto & superCell0 = pb0.getSuperCell( superCellIdx );
            uint32_t numParticlesInSupercell0 = superCell0.getNumParticles();

            auto & superCell1 = pb1.getSuperCell( superCellIdx );
            uint32_t numParticlesInSupercell1 = superCell1.getNumParticles();


            /* loop over all particles in the frame */
            ForEachIdx< FrameDomCfg > forEachFrameElem( workerIdx );

            // ###### species 0
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

            FramePtr0 firstFrame0 = pb0.getFirstFrame( superCellIdx );
            detail::particlesCntHistogram( acc, forEachFrameElem, pb0, firstFrame0, numParticlesInSupercell0, nppc );

            __syncthreads();

            // memory for particle indices
            forEachFrameElem(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const
                )
                {
                    parCellList0[ linearIdx ].init( deviceHeapHandle,  nppc[ linearIdx ] );
                }
            );

            __syncthreads();

            detail::updateLinkedList( acc, forEachFrameElem, pb0, firstFrame0, numParticlesInSupercell0, parCellList0 );

            __syncthreads();
#if 1
            if(threadIdx.x == 0)
            {
                for(int i=0;i<256;++i)
                    if( nppc[ i ] != parCellList0[ i ].size)
                        printf("ppc %i: %u == %u\n",i, nppc[ i ], parCellList0[ i ].size);
            }
            __syncthreads();
#endif

            //######## species 1

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

            FramePtr1 firstFrame1 = pb1.getFirstFrame( superCellIdx );
            detail::particlesCntHistogram( acc, forEachFrameElem, pb1, firstFrame1, numParticlesInSupercell1, nppc );

            __syncthreads();

            // memory for particle indices
            forEachFrameElem(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const
                )
                {
                    parCellList1[ linearIdx ].init( deviceHeapHandle,  nppc[ linearIdx ] );
                }
            );

            __syncthreads();

            detail::updateLinkedList( acc, forEachFrameElem, pb1, firstFrame1, numParticlesInSupercell1, parCellList1 );

            __syncthreads();
#if 1
            if(threadIdx.x == 0)
            {
                for(int i=0;i<256;++i)
                    if( nppc[ i ] != parCellList1[ i ].size)
                        printf("ppc %i: %u == %u\n",i, nppc[ i ], parCellList1[ i ].size);
            }
            __syncthreads();
#endif




            //shuffle indices list of the longest particle list
            forEachFrameElem(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const idx
                )
                {
                    //find longer list
                    auto * longParList = parCellList0[linearIdx].size >= parCellList1[linearIdx].size ? &parCellList0[linearIdx] : &parCellList1[linearIdx];
                    (*longParList).shuffle( acc, rngHandle );
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
                    uint32_t const
                )
                {

                    uint32_t const numParPerCellInShortList = math::min(parCellList0[linearIdx].size, parCellList1[linearIdx].size);

                    // skip particle offset counter
                    uint32_t* parListStart0 = parCellList0[ linearIdx ].ptrToIndicies;
                    uint32_t* parListStart1 = parCellList1[ linearIdx ].ptrToIndicies;


                    for(uint32_t i = 0; i < numParPerCellInShortList; ++i)
                    {
#if 1
                        if(parListStart0[ i ] >= numParticlesInSupercell0 || parListStart1[ i ] >= numParticlesInSupercell1)
                            printf("wrong %u/%u %u/%u\n", parListStart0[ i ],numParticlesInSupercell0,parListStart1[ i ], numParticlesInSupercell1 );
#endif
                        auto srcPar = getParticle(pb0, firstFrame0, parListStart0[ i ]);
                        auto destPar = getParticle(pb1, firstFrame1, parListStart1[ i ]);
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
                    parCellList0[ linearIdx ].finalize( deviceHeapHandle );
                    parCellList1[ linearIdx ].finalize( deviceHeapHandle );
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
        typename T_Species0,
        typename T_Species1
    >
    struct DoInterCollision
    {
        void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap, uint32_t currentStep)
        {
            using Species0 = T_Species0;
            using FrameType0 = typename Species0::FrameType;

            using Species1 = T_Species1;
            using FrameType1 = typename Species1::FrameType;

            using CollisionFunctor = T_CollisionFunctor;

            DataConnector &dc = Environment<>::get().DataConnector();
            auto species0 = dc.get< Species0 >( FrameType0::getName(), true );
            auto species1 = dc.get< Species1 >( FrameType1::getName(), true );

            // use mapping information from the first species
            AreaMapping<
                CORE + BORDER,
                picongpu::MappingDesc
            > mapper( species0->getCellDescription() );

            constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                pmacc::math::CT::volume< SuperCellSize >::type::value
            >::value;

            /* random number generator */
            using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;

            PMACC_KERNEL( InterCollision< numWorkers >{ } )(
                mapper.getGridDim(),
                numWorkers
            )(
                species0->getDeviceParticlesBox( ),
                species1->getDeviceParticlesBox( ),
                mapper,
                deviceHeap->getAllocatorHandle(),
                RNGFactory::createHandle(),
                CollisionFunctor(currentStep)
            );

            dc.releaseData( FrameType0::getName() );
            dc.releaseData( FrameType1::getName() );
        }
    };

} // namespace collision
} // namespace particles
} // namespace picongpu

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
namespace picongpu
{

namespace particles
{

    DINLINE void swap( uint32_t & v0, uint32_t & v1)
    {
        uint32_t tmp = v0;
        v0 = v1;
        v1 = tmp;
    }

    template<
        typename T_SrcPar,
        typename T_DestPar
    >
    DINLINE void collision( uint32_t rn, float_X s, float_X cellDensity, T_SrcPar& srcPar, T_DestPar& destPar)
    {
        float_X rngValue = float_X(rn%8096)/8096._X;
        auto srcParMom = srcPar[ momentum_ ];
        auto destParMom = destPar[ momentum_ ];

        srcPar[ momentum_ ] = srcParMom * rngValue + destParMom * ( 1.0_X * rngValue );
        destPar[ momentum_ ] = destParMom * rngValue + srcParMom * ( 1.0_X * rngValue );
    }

    template< uint32_t T_numWorkers >
    struct Collision
    {
        template<
            typename T_ParBox,
            typename T_Mapping,
            typename T_Acc,
            typename T_DeviceHeapHandle,
            typename T_Rng
        >
        DINLINE void operator()(
            T_Acc const & acc,
            T_ParBox pb,
            T_Mapping const mapper,
            T_DeviceHeapHandle deviceHeapHandle,
            T_Rng rng
        ) const
        {
            using namespace pmacc::particles::operations;
            using namespace mappings::threads;

            constexpr uint32_t frameSize = pmacc::math::CT::volume<typename T_Mapping::SuperCellSize>::type::value;
            constexpr uint32_t dim = T_Mapping::Dim;
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
                parListPtr,
                memory::Array<
                    uint32_t*,
                    frameSize
                >
            );

            uint32_t const workerIdx = threadIdx.x;

            using MasterOnly = IdxConfig<
                1,
                numWorkers
            >;

            using FrameDomCfg = IdxConfig<
                frameSize,
                numWorkers
            >;

            DataSpace< dim > const superCellIdx = mapper.getSuperCellIndex( DataSpace< dim > ( blockIdx ) );

            // offset of the superCell (in cells, without any guards) to the origin of the local domain
            DataSpace< simDim > const localSuperCellOffset =
            superCellIdx - mapper.getGuardingSuperCells( );
            rng.init(
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

            FramePtr frame = pb.getFirstFrame( superCellIdx );

            // histogram
            for(uint32_t i = 0; i < numParticlesInSupercell; i += frameSize)
            {

                forEachFrameElem(
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
                frame = pb.getNextFrame( frame );
            }

            __syncthreads();

            // memory for particle indices
            forEachFrameElem(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const idx
                )
                {
                   // printf("alloc %u: %u\n", linearIdx, (nppc[ linearIdx ] + 1) );
                #if( PMACC_CUDA_ENABLED == 1 )
                    parListPtr[ linearIdx ] = nullptr;
                    while( parListPtr[ linearIdx ] == nullptr )
                        parListPtr[ linearIdx ] = (uint32_t*) deviceHeapHandle.malloc( sizeof(uint32_t) * ( nppc[ linearIdx ] + 1 ) );
                #else
                    parListPtr[ linearIdx ] = new uint32_t[ nppc[ linearIdx ] + 1 ];
                #endif
                    //reset counter
                    parListPtr[ linearIdx ][0] = 0u;
                }
            );

            __syncthreads();


            frame = pb.getFirstFrame( superCellIdx );
            // fill indices list
            for(uint32_t i = 0; i < numParticlesInSupercell; i += frameSize)
            {

                forEachFrameElem(
                    [&](
                        uint32_t const linearIdx,
                        uint32_t const idx
                    )
                    {
                        if( i + linearIdx < numParticlesInSupercell )
                        {
                            auto particle = frame[ linearIdx ];
                            auto parLocalIndex = particle[ localCellIdx_ ];
                            uint32_t parOffset = atomicAdd( parListPtr[ parLocalIndex ], 1u );
                            // start at first storage because index 0 is the offset counter
                            parListPtr[ parLocalIndex ][ parOffset + 1 ] = i + linearIdx;
                        }
                    }
                );
                frame = pb.getNextFrame( frame );
            }

            __syncthreads();
#if 1
            if(threadIdx.x == 0)
            {
                for(int i=0;i<256;++i)
                    if( nppc[ i ] != parListPtr[ i ][0])
                        printf("ppc %i: %u == %u\n",i, nppc[ i ], parListPtr[ i ][0]);
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
                    uint32_t const numParPerCell = nppc[ linearIdx ];
                    uint32_t* parListStart = parListPtr[ linearIdx ] + 1;
                    // shuffle the particle lookup table
                    for(uint32_t i = numParPerCell; i > 0; --i)
                    {
                        /* modulo is not perfect but okish,
                         * because of the loop head mod zero is not possible
                         */
                        int p = rng(acc) % i;
                        if( i - 1 != p )
                            swap(parListStart[ i - 1 ], parListStart[ p ]);
                    }
                }
            );

            auto firstFrame = pb.getFirstFrame( superCellIdx );
            forEachFrameElem(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const idx
                )
                {
                    uint32_t const numParPerCell = nppc[ linearIdx ];

                    // skip particle offset counter
                    uint32_t* parListStart = parListPtr[ linearIdx ] + 1;

#if 0
                    auto const fieldOffset = localDomainOffset +
                        DataSpaceOperations< simDim >::map(
                            SuperCellSize::toRT(),
                            linearIdx
                        ) +
                        SuperCellSize::toRT() * GuardSize::toRT();

                    float_X temperature = temperatureBox(fieldOffset );
                    float_X cellDensity = densityBox(fieldOffset );
#endif
                    if(numParPerCell != 0)
                        for(uint32_t i = 0; i < numParPerCell - 1u; i += 2)
                        {
#if 1
                            if(parListStart[ i ] >= numParticlesInSupercell || parListStart[ i + 1 ] >= numParticlesInSupercell)
                                printf("wrong %u %u of %u\n", parListStart[ i ],parListStart[ i  + 1],numParPerCell );
#endif
                            auto srcPar = getParticle(pb, firstFrame, parListStart[ i ]);
                            auto destPar = getParticle(pb, firstFrame, parListStart[ i + 1 ]);
                            collision(rng(acc), 0._X, 0._X, srcPar, destPar);
                        }
                }
            );

            forEachFrameElem(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const idx
                )
                {
#if( PMACC_CUDA_ENABLED == 1 )
                    deviceHeapHandle.free( (void*) parListPtr[ linearIdx ] );
#else
                    delete(parListPtr[ linearIdx ]);
#endif
                    parListPtr[ linearIdx ] = nullptr;
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

    template<typename T_SpeciesType>
    struct DoCollision
    {
        using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<
            VectorAllSpecies,
            T_SpeciesType
        >;
        using FrameType = typename SpeciesType::FrameType;

        void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap)
        {
            DataConnector &dc = Environment<>::get().DataConnector();
            auto species = dc.get< SpeciesType >( FrameType::getName(), true );

            AreaMapping<
                CORE + BORDER,
                picongpu::MappingDesc
            > mapper( species->getCellDescription() );

            constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                pmacc::math::CT::volume< SuperCellSize >::type::value
            >::value;

            /* random number generator */
            using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
            using Distribution = pmacc::random::distributions::Uniform<uint32_t>;

            PMACC_KERNEL( Collision< numWorkers >{ } )(
                mapper.getGridDim(),
                numWorkers
            )(
                species->getDeviceParticlesBox( ),
                mapper,
                deviceHeap->getAllocatorHandle(),
                RNGFactory::createRandom<Distribution>()
            );

            dc.releaseData( FrameType::getName() );
        }
    };

} // namespace particles
} // namespace picongpu

/* Copyright 2019 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once


namespace pmacc
{
namespace algorithms
{
namespace kernel
{

    struct Histogram
    {
        constexpr uint32_t frameSize = math::CT::volume<typename T_Mapping::SuperCellSize>::type::value;
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

        // data copied from right (last) to left (first)
        PMACC_SMEM(
            acc,
            firstFrame,
            FramePtr
        );

        uint32_t const workerIdx = threadIdx.x;

        using MasterOnly = IdxConfig<
            1,
            numWorkers
        >;

        using ParticleDomCfg = IdxConfig<
            frameSize,
            numWorkers
        >;


        DataSpace< dim > const superCellIdx = mapper.getSuperCellIndex( DataSpace< dim > ( blockIdx ) );

        auto & superCell = pb.getSuperCell( superCellIdx );
        uint32_t numParticlesInSupercell = superCell.getNumParticles();

        onlyMaster(
            [&](
                uint32_t const,
                uint32_t const
            )
            {
                firstFrame = pb.getFirstFrame( superCellIdx );
            }
        );

        /* loop over all particles in the frame */
        ForEachIdx< ParticleDomCfg > forEachParticle( workerIdx );

        forEachParticle(
            [&](
                uint32_t const linearIdx,
                uint32_t const
            )
            {
                nppc[ linearIdx ] = 0u;
            }
        );

        __syncthreads();

        for(uint32_t i = 0; i < numParticlesInSupercell; i += frameSize)
        {

            forEachParticle(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const idx
                )
                {
                    if( i + linearIdx < numParticlesInSupercell)
                    {
                        auto & particle = lastFrame[ linearIdx ];
                        auto parLocalIndex = particle[ localCellIdx_ ];
                        atomicAdd( &nppc[ parLocalIndex ], 1u);
                    }
                }
            );
            firstFrame = pb.getNextFrame( firstFrame );
        }


    };

} // namespace kernel
} // namespace algorithms
} // namespace pmacc

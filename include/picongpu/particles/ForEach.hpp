/* Copyright 2014-2018 Rene Widera
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
#include "picongpu/particles/filter/filter.def"
#include "picongpu/particles/manipulators/manipulators.def"

#include <pmacc/Environment.hpp>
#include <pmacc/particles/compileTime/FindByNameOrType.hpp>

#include <boost/mpl/apply.hpp>

#include "picongpu/simulation_defines.hpp"
#include <pmacc/particles/frame_types.hpp>
#include <pmacc/particles/memory/boxes/ParticlesBox.hpp>
#include <pmacc/particles/memory/boxes/TileDataBox.hpp>

#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/FieldB.hpp"

#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/CachedBox.hpp>

#include <pmacc/nvidia/functors/Assign.hpp>
#include <pmacc/mappings/threads/ThreadCollective.hpp>

#include <pmacc/nvidia/rng/RNG.hpp>
#include <pmacc/nvidia/rng/methods/Xor.hpp>
#include <pmacc/nvidia/rng/distributions/Normal_float.hpp>

#include <pmacc/particles/operations/Assign.hpp>
#include <pmacc/particles/operations/Deselect.hpp>
#include <pmacc/nvidia/atomic.hpp>
#include "picongpu/particles/InterpolationForPusher.hpp"
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/mappings/threads/ForEachIdx.hpp>
#include <pmacc/mappings/threads/IdxConfig.hpp>
#include <pmacc/mappings/threads/WorkerCfg.hpp>


namespace picongpu
{
namespace particles
{

    /** manipulate particles of a species
     *
     * This functor prepares a particle box to call
     * a user defined unary functor which allows to manipulate the particles.
     *
     * @tparam T_numWorkers number of workers
     */
    template< uint32_t T_numWorkers >
    struct KernelForEach
    {
        /** frame-wise manipulate particles
         *
         * @tparam T_ParBox pmacc::ParticlesBox, type of the species box
         * @tparam T_ManipulateFunctor type of the user functor to derive a particle
         * @tparam T_Mapping mapping functor type
         *
         * @param pb particles to manipulate
         * @param manipulateFunctor functor to manipulate a particle
         *                          must fulfill the interface particles::manipulators::IUnary
         * @param mapper functor to map a block to a supercell
         */
        template<
            typename T_ManipulateFunctor,
            typename T_ParBox,
            typename T_Mapping,
            typename T_Acc
        >
        DINLINE void operator()(
            T_Acc const & acc,
            T_ParBox pb,
            T_ManipulateFunctor manipulateFunctor,
            T_Mapping const mapper
        ) const
        {
            using namespace mappings::threads;

            constexpr uint32_t frameSize = pmacc::math::CT::volume< SuperCellSize >::type::value;
            constexpr uint32_t numWorkers = T_numWorkers;

            uint32_t const workerIdx = threadIdx.x;

            using FramePtr = typename T_ParBox::FramePtr;
            PMACC_SMEM(
                acc,
                frame,
                FramePtr
            );

            DataSpace< simDim > const superCellIdx(
                mapper.getSuperCellIndex( DataSpace< simDim >( blockIdx ) )
            );

            ForEachIdx<
                IdxConfig<
                    1,
                    numWorkers
                >
            > onlyMaster{ workerIdx };

            onlyMaster(
                [&](
                    uint32_t const,
                    uint32_t const
                )
                {
                    frame = pb.getLastFrame( superCellIdx );
                }
            );

            __syncthreads();

            // end kernel if we have no frames
            if( !frame.isValid( ) )
                return;

            using ParticleDomCfg = IdxConfig<
                frameSize,
                numWorkers
            >;

            // marker if a particle slot within a frame holds a valid particle
            memory::CtxArray<
                bool,
                ParticleDomCfg
            >
            isParticleCtx(
                workerIdx,
                [&](
                    uint32_t const linearIdx,
                    uint32_t const
                )
                {
                    return frame[ linearIdx ][ multiMask_ ];
                }
            );

            // offset of the superCell (in cells, without any guards) to the origin of the local domain
            DataSpace< simDim > const localSuperCellOffset =
                superCellIdx - mapper.getGuardingSuperCells( );

            auto accManipulator = manipulateFunctor(
                acc,
                localSuperCellOffset,
                WorkerCfg< T_numWorkers >{ workerIdx }
            );

            __syncthreads( );

            while( frame.isValid( ) )
            {
                // loop over all particles in the frame
                ForEachIdx< ParticleDomCfg >{ workerIdx }(
                    [&](
                        uint32_t const linearIdx,
                        uint32_t const idx
                    )
                    {
                        auto particle = frame[ linearIdx ];
                        bool const isPar = isParticleCtx[ idx ];

                        if( !isPar )
                            particle.setHandleInvalid( );

                        // call manipulator even if the particle is not valid
                        accManipulator( acc, particle );

                        /* only the last frame is allowed to be non-full: all following
                         * frames' particles will be valid, since we iterate the list of
                         * frames from back to front
                         */
                        isParticleCtx[ idx ] = true;
                    }
                );

                __syncthreads( );

                onlyMaster(
                    [&](
                        uint32_t const,
                        uint32_t const
                    )
                    {
                        frame = pb.getPreviousFrame( frame );
                    }
                );

                __syncthreads( );
            }
        }
    };

    /** Run a user defined manipulation for each particle of a species
     *
     * Allows to manipulate attributes of existing particles in a species with
     * arbitrary unary functors ("manipulators").
     *
     * @warning Does NOT call FillAllGaps after manipulation! If the
     *          manipulation deactivates particles or creates "gaps" in any
     *          other way, FillAllGaps needs to be called for the
     *          `T_SpeciesType` manually in the next step!
     *
     * @tparam T_Manipulator unary lambda functor accepting one particle
     *                       species,
     *                       @see picongpu::particles::manipulators
     * @tparam T_SpeciesType type or name as boost::mpl::string of the used species
     * @tparam T_Filter picongpu::particles::filter, particle filter type to
     *                  select particles in `T_SpeciesType` to manipulate via
     *                  `T_DestSpeciesType`
     */
    template<
        typename T_Manipulator,
        typename T_SpeciesType = bmpl::_1,
        typename T_Filter = filter::All
    >
    struct ForEachParticle
    {
        template< typename T_Mapper >
        HINLINE void
        operator()(
            uint32_t const currentStep,
            T_Mapper mapper
        )
        {

            using SpeciesType = pmacc::particles::compileTime::FindByNameOrType_t<
                VectorAllSpecies,
                T_SpeciesType
            >;
            using FrameType = typename SpeciesType::FrameType;

            using SpeciesFunctor = typename bmpl::apply1<
                T_Manipulator,
                SpeciesType
            >::type;

            using FilteredManipulator = manipulators::IUnary<
                SpeciesFunctor,
                T_Filter
            >;

            DataConnector &dc = Environment<>::get().DataConnector();
            auto speciesPtr = dc.get< SpeciesType >(
                FrameType::getName(),
                true
            );

            FilteredManipulator filteredManipulator( currentStep );

            constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                pmacc::math::CT::volume< SuperCellSize >::type::value
            >::value;

            PMACC_KERNEL( KernelForEach< numWorkers >{ } )(
                mapper.getGridDim( ),
                numWorkers
            )(
                speciesPtr->getDeviceParticlesBox( ),
                filteredManipulator,
                mapper
            );

            dc.releaseData( FrameType::getName() );
        }
    };

} //namespace particles
} //namespace picongpu

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
#include "picongpu/particles/collision/InterSpecies.hpp"


namespace picongpu
{
namespace particles
{
namespace collision
{
namespace detail
{

    template<
        typename T_CollisionFunctor,
        typename T_BaseSpecies,
        typename T_PeerSpecies
    >
    struct WithPeer
    {
        void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap, uint32_t currentStep)
        {
            DoInterCollision<
                T_CollisionFunctor,
                T_BaseSpecies,
                T_PeerSpecies
            >{}(
                deviceHeap,
                currentStep
            );
        }
    };

    template<
        typename T_CollisionFunctor,
        typename T_Species
    >
    struct WithPeer< T_CollisionFunctor, T_Species, T_Species >
    {
        void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap, uint32_t currentStep)
        {
            DoIntraCollision<
                T_CollisionFunctor,
                T_Species
            >{}(
                deviceHeap,
                currentStep
            );
        }
    };
} // namespace detail

    template<
        typename T_CollisionFunctor,
        typename T_BaseSpecies,
        typename T_PeerSpecies,
        typename T_Filter = filter::All
    >
    struct WithPeer
    {
        void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap, uint32_t currentStep)
        {
            using BaseSpecies = pmacc::particles::meta::FindByNameOrType_t<
                VectorAllSpecies,
                T_BaseSpecies
            >;

            using PeerSpecies = pmacc::particles::meta::FindByNameOrType_t<
                VectorAllSpecies,
                T_PeerSpecies
            >;

            using CollisionFunctor = typename bmpl::apply2<
                T_CollisionFunctor,
                BaseSpecies,
                PeerSpecies
            >::type;
            using FilteredCollisionFunctor = IBinary<
                CollisionFunctor,
                T_Filter
            >;

            detail::WithPeer<
                FilteredCollisionFunctor,
                BaseSpecies,
                PeerSpecies
            >{}(
                deviceHeap,
                currentStep
            );
        }
    };

} // namespace collision
} // namespace particles
} // namespace picongpu

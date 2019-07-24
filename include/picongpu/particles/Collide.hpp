/* Copyright 2019 Rene Widera
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
#include "picongpu/particles/traits/GetColliderList.hpp"
#include "picongpu/particles/Collide.def"
#include "picongpu/particles/collision/WithPeer.hpp"

#include <pmacc/meta/conversion/ApplyGuard.hpp>
#include <pmacc/meta/conversion/ToSeq.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <boost/mpl/apply.hpp>


namespace picongpu
{
namespace particles
{

namespace detail
{
    template<
        typename T_Collider,
        typename T_BaseSpecies
    >
    struct Collide
    {
        void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap, uint32_t currentStep)
        {
            using PeerSpeciesList = typename pmacc::ToSeq< typename T_Collider::Species >::type;

            meta::ForEach<
                PeerSpeciesList,
                collision::WithPeer<
                    ApplyGuard< typename T_Collider::Functor >,
                    T_BaseSpecies,
                    bmpl::_1,
                    typename T_Collider::Filter
                >
            >{}( deviceHeap, currentStep );
        }
    };

} // namespace detail

    template<
        typename T_Species = bmpl::_1
    >
    struct Collide
    {
        void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap, uint32_t currentStep)
        {
            using Species = pmacc::particles::meta::FindByNameOrType_t<
                VectorAllSpecies,
                T_Species
            >;
            using ColliderList = typename traits::GetColliderList< Species >::type;

            meta::ForEach<
                ColliderList,
                detail::Collide<
                    bmpl::_1,
                    Species
                >
            >{}( deviceHeap, currentStep );
        }
    };

} //namespace particles
} //namespace picongpu

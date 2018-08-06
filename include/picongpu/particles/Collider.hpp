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

#include "picongpu/particles/Manipulate.def"

#include <pmacc/compileTime/ApplyGuard.hpp>

#include <boost/mpl/apply.hpp>
#include <boost/mpl/placeholders.hpp>


namespace picongpu
{
namespace particles
{
    template<
        typename T_CollisionAlgorithm,
        typename T_PairSpeciesType,
        typename T_Filter = filter::All
    >
    using Collider = Manipulate<
        // avoid that the target species is applied to the collision algorithm
        compileTime::ApplyGuard<
            typename bmpl::apply1<
                T_CollisionAlgorithm,
                T_PairSpeciesType
            >::type
        >,
        bmpl::_1,
        T_Filter
    >;

    template<
        typename T_ColliderSeq,
        typename T_FinilizerSeq = bmpl::vector0<>
    >
    struct Collision
    {
        using ColliderSeq = T_ColliderSeq;
        using FinilizerSeq = T_FinilizerSeq;
    };

} //namespace particles
} //namespace picongpu

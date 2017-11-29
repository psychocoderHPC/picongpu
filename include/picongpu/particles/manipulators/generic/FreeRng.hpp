/* Copyright 2015-2017 Rene Widera, Alexander Grund
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
#include "picongpu/particles/manipulators/generic/FreeRng.def"
#include "picongpu/particles/functor/generic/FreeRng.hpp"

#include <utility>
#include <type_traits>
#include <string>


namespace picongpu
{
namespace particles
{
namespace manipulators
{
namespace generic
{
    template<
        typename T_Functor,
        typename T_Distribution,
        typename T_Seed,
        typename T_SpeciesType
    >
    struct FreeRng : public functor::generic::FreeRng<
        T_Functor,
        T_Distribution,
        T_Seed,
        T_SpeciesType
    >
    {
        using Base = functor::generic::FreeRng<
            T_Functor,
            T_Distribution,
            T_Seed,
            T_SpeciesType
        >;

        /** constructor
         *
         * T_Functor can only have one constructor of the following constructors:
         * T_Functor( currentStep ) or the default constructor T_Functor()
         *
         * @param currentStep current simulation time step
         */
        HINLINE FreeRng(
            uint32_t currentStep
        ) : Base( currentStep )
        {
        }
    };

} // namepsace generic
} // namespace manipulators
} // namespace particles
} // namespace picongpu

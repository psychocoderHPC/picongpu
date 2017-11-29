/* Copyright 2013-2017 Rene Widera, Axel Huebl
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

#include "picongpu/particles/filter/generic/Free.def"
#include "picongpu/particles/functor/generic/Free.hpp"

#include <utility>
#include <type_traits>

namespace picongpu
{
namespace particles
{
namespace filter
{
namespace generic
{

    template< typename T_Functor >
    struct Free : public functor::generic::Free< T_Functor >
    {
        using Base = functor::generic::Free< T_Functor >;

        /** constructor
         *
         * T_Functor can only have one constructor of the following constructors:
         * T_Functor( currentStep ) or the default constructor T_Functor()
         *
         * @param currentStep current simulation time step
         */
        HINLINE Free( uint32_t currentStep ) : Base( currentStep )
        {
        }
    };

} // namespace generic
} // namespace filter
} // namespace particles
} // namespace picongpu

/* Copyright 2016-2022 Rene Widera
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

namespace picongpu
{
    namespace currentSolver
    {
        template<typename T_ParticleAssignFunctor>
        struct GitShape
        {
            float_X const particlePosition;
            HDINLINE GitShape(float_X const position, bool const /*isInBaseAssignmentCell*/)
                : particlePosition(position)
            {
            }

            HDINLINE GitShape(float_X const position)
                : particlePosition(position)
            {
            }

            HDINLINE float_X operator()(int const gridPointOffset) const
            {
                return T_ParticleAssignFunctor()(float_X(gridPointOffset) - particlePosition);
            }
        };

        template<typename T_ParticleAssignFunctor>
        struct CachedShape
        {
            HDINLINE CachedShape(float_X const position, bool const isInBaseAssignmentCell)
                : shapeArray(std::move(T_ParticleAssignFunctor().shapeArray(position, !isInBaseAssignmentCell)))
            {
            }

            HDINLINE float_X operator()(int const gridPointOffset) const
            {
                return shapeArray[gridPointOffset - T_ParticleAssignFunctor::begin];
            }

            decltype(T_ParticleAssignFunctor().shapeArray(
                alpaka::core::declval<float_X>(),
                alpaka::core::declval<bool>())) shapeArray;
        };

    } // namespace currentSolver
} // namespace picongpu

/* Copyright 2021-2022 Sergei Bastrakov
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


namespace picongpu::fields::maxwellSolver
{
    /** Check the CFL condition
     *
     * @tparam T_FieldSolver field solver type
     * @return value of 'X' to fulfill the condition 'c * dt <= X`
     */
    template<typename T_FieldSolver>
    inline float_X checkCfl(T_FieldSolver const&)
    {
        static_assert(
            sizeof(T_FieldSolver) && false,
            "checkCfl() not implemented for the current selected field solver.");
        return 0.0_X;
    }
} // namespace picongpu::fields::maxwellSolver

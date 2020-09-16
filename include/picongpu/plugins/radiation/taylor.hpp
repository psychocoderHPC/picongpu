/* Copyright 2013-2020 Heiko Burau, Rene Widera, Richard Pausch
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

#include "VectorTypes.hpp"

namespace picongpu
{
namespace plugins
{
namespace radiation
{
struct Taylor
{
    // a Taylor development for 1-sqrt(1-x)

    HDINLINE picongpu::float_X operator()(picongpu::float_X x) const
    {
        // Taylor series of 1-sqrt(1-x) till 5th order
        //same like 0.5*x + 0.125*x*x + 0.0625 * x*x*x + 0.0390625 * x*x*x*x + 0.02734375 *x*x*x*x*x;
        const picongpu::float_X x2 = (x * x);
        return x * ((0.5_X + 0.125_X * x) + x2 * (0.0625_X + (0.0390625_X * x + 0.02734375_X * x2)));
    }

};

} // namespace radiation
} // namespace plugins
} // namespace picongpu

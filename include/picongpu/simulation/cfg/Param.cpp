/* Copyright 2023 Rene Widera
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

#include "picongpu/simulation/cfg/Param.hpp"

#include "picongpu/param/dimension.param"
#include "picongpu/param/precision.param"
#include "picongpu/simulation/cfg/Simulation.hpp"

#include <pmacc/algorithms/TypeCast.hpp>
#include <pmacc/types.hpp>


namespace picongpu::simulation::cfg
{
    template<typename T_FloatType>
    HINLINE void Param<T_FloatType>::convertToPic(Simulation& sim)
    {
        auto const& si = sim.si;
        auto const& siUnit = sim.si.unit;

        delta_t = static_cast<T_FloatType>(si.delta_t / siUnit.time);
        cell = precisionCast<T_FloatType>(si.cell / siUnit.length);
        particle.num_per_cell = si.particle.num_per_cell;
        particle.density
            = static_cast<T_FloatType>(si.particle.density * siUnit.length * siUnit.length * siUnit.length);
    }

    template struct Param<float>;
    template struct Param<double>;

} // namespace picongpu::simulation::cfg

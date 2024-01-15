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

#include "picongpu/simulation/cfg/Base.hpp"

#include "picongpu/simulation/cfg/Simulation.hpp"


namespace picongpu::simulation::cfg
{
    template<typename T_FloatType>
    HINLINE void Base<T_FloatType>::updateSi(Simulation& sim)
    {
        particle.mass = sim.si.physicalConstant.electron.mass;
        particle.charge = sim.si.physicalConstant.electron.charge;
        particle.typical_num_particles_per_macroparticle = (particle.density * sim.si.cell.productOfComponents())
            / static_cast<T_FloatType>(sim.si.particle.num_per_cell);
        particle.density = sim.si.particle.density;
    }

    template<typename T_FloatType>
    HINLINE void Base<T_FloatType>::convertToPic(Simulation& sim)
    {
        auto const& siBasePar = sim.si.base.particle;
        auto const& siUnit = sim.si.unit;

        particle.mass = static_cast<T_FloatType>(siBasePar.mass / siUnit.mass);
        particle.charge = static_cast<T_FloatType>(siBasePar.charge / siUnit.charge);
        particle.typical_num_particles_per_macroparticle = siBasePar.typical_num_particles_per_macroparticle;
        particle.density = static_cast<T_FloatType>(siBasePar.density * siUnit.length * siUnit.length * siUnit.length);
    }

    template struct Base<float>;
    template struct Base<double>;
} // namespace picongpu::simulation::cfg

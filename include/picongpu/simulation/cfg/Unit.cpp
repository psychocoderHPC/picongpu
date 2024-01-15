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

#include "picongpu/simulation/cfg/Unit.hpp"

#include "picongpu/simulation/cfg/Simulation.hpp"


namespace picongpu::simulation::cfg
{
    template<typename T_FloatType>
    HINLINE void Unit<T_FloatType>::updateSi(Simulation& sim)
    {
        speed = sim.si.physicalConstant.speed_of_light;
        time = sim.si.delta_t;
        length = time * speed;
        mass = sim.si.base.particle.mass * sim.si.base.particle.typical_num_particles_per_macroparticle;
        charge = 1.0 * sim.si.base.particle.charge * sim.si.base.particle.typical_num_particles_per_macroparticle;
        energy = mass * length * length / (time * time);
        efield = 1.0 / (time * time / mass / length * charge);
        bfield = mass / (time * charge);
    }

    template<typename T_FloatType>
    HINLINE void Unit<T_FloatType>::convertToPic(Simulation& sim)
    {
        auto const& siUnit = sim.si.unit;

        speed = static_cast<T_FloatType>(siUnit.speed);
        time = static_cast<T_FloatType>(siUnit.time);
        length = static_cast<T_FloatType>(siUnit.length);
        mass = static_cast<T_FloatType>(siUnit.mass);
        charge = static_cast<T_FloatType>(siUnit.charge);
        energy = static_cast<T_FloatType>(siUnit.energy);
        efield = static_cast<T_FloatType>(siUnit.efield);
        bfield = static_cast<T_FloatType>(siUnit.bfield);

        atomic.time =static_cast<T_FloatType>(siUnit.atomic.time / siUnit.time);
        atomic.energy =static_cast<T_FloatType>(siUnit.atomic.energy / siUnit.energy);
        atomic.efield =static_cast<T_FloatType>(siUnit.atomic.efield / siUnit.efield);
    }

    template struct Unit<float>;
    template struct Unit<double>;
} // namespace picongpu::simulation::cfg

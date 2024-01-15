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
    HINLINE void PhysicalConstant<T_FloatType>::updateSi(Simulation& sim)
    {
        eps0 = 1.0 / mue0 / speed_of_light / speed_of_light;
        z0 = mue0 * speed_of_light;
        electron.radius
            = electron.charge * electron.charge / (4.0 * PI * eps0 * electron.mass * speed_of_light * speed_of_light);
        mue0_eps0 = 1.0 / speed_of_light / speed_of_light;
    }

    template<typename T_FloatType>
    HINLINE void PhysicalConstant<T_FloatType>::convertToPic(Simulation& sim)
    {
        auto const& siPhyConst = sim.si.physicalConstant;
        auto const& siUnit = sim.si.unit;

        speed_of_light = static_cast<T_FloatType>(siPhyConst.speed_of_light / siUnit.speed);
        mue0 = static_cast<T_FloatType>(siPhyConst.mue0 / siUnit.length / siUnit.mass * siUnit.charge * siUnit.charge);
        eps0 = static_cast<T_FloatType>(1.0 / mue0 / siPhyConst.speed_of_light / siPhyConst.speed_of_light);
        mue0_eps0 = static_cast<T_FloatType>(siPhyConst.mue0 / siPhyConst.speed_of_light / siPhyConst.speed_of_light);
        z0 = mue0 * speed_of_light;
        hbar = static_cast<T_FloatType>(siPhyConst.hbar / siUnit.energy / siUnit.time);
        electron.charge = static_cast<T_FloatType>(siPhyConst.electron.charge / siUnit.charge);
        electron.mass = static_cast<T_FloatType>(siPhyConst.electron.mass / siUnit.mass);
        electron.radius = static_cast<T_FloatType>(siPhyConst.electron.radius / siUnit.length);
    }

    template struct PhysicalConstant<float>;
    template struct PhysicalConstant<double>;
} // namespace picongpu::simulation::cfg

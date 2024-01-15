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

#pragma once

#include <pmacc/types.hpp>

namespace picongpu::simulation::cfg
{
    struct Simulation;

    template<typename T_FloatType>
    struct Unit
    {
        T_FloatType speed;
        T_FloatType time;
        T_FloatType length;
        T_FloatType mass;
        T_FloatType charge;
        T_FloatType energy;
        T_FloatType efield;
        T_FloatType bfield;

        struct Atomic
        {
            /* atomic unit for energy:
             * 1 Rydberg = 27.21 eV --> converted to Joule
             */
            T_FloatType energy = 4.36e-18;
            /* atomic unit for time in s:
             * 150 attoseconds (classical electron orbit time in hydrogen)  / 2 PI
             */
            T_FloatType time = 2.4189e-17;
            /* atomic unit for electric field in V/m:
             * field strength between electron and core in ground state hydrogen
             */
            T_FloatType efield = 5.14e11;
        };
        Atomic atomic;

        void updateSi(Simulation& sim);
        void convertToPic(Simulation& sim);
    };
} // namespace picongpu::simulation::cfg

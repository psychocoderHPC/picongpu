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
    struct Conversion
    {
        T_FloatType keV_to_joule = 1.60217646e-16;
        T_FloatType ev_to_joule;
        T_FloatType joul_to_eV;
        T_FloatType joule_to_keV;
        T_FloatType joule_to_eV;

        struct Atomic
        {
            /* 1 atomic unit of energy is equal to 1 Hartree or 2 Rydberg
             * which is twice the ground state binding energy of atomic hydrogen */
            T_FloatType au_to_ev = 27.21139;
            T_FloatType ev_to_au;
        };
        Atomic atomic;

        void updateSi(Simulation& sim);
        void convertToPic(Simulation& sim);
    };

} // namespace picongpu::simulation::cfg

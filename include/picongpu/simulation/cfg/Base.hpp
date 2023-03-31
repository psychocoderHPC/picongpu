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
    struct Base
    {

        struct Particle
        {
            T_FloatType typical_num_particles_per_macroparticle;
            T_FloatType density;
            /** unit: kg */
            T_FloatType charge;
            /** unit: C */
            T_FloatType mass;
        };
        Particle particle;

        void updateSi(Simulation& sim);
        void convertToPic(Simulation& sim);
    };
} // namespace picongpu::simulation::cfg

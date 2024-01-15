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

#include "picongpu/param/dimension.param"
#include "picongpu/param/precision.param"

#include <pmacc/types.hpp>

namespace picongpu::simulation::cfg
{
    struct Simulation;

    template<typename T_FloatType>
    struct Param
    {
        //Param() : cell(vec3<T_FloatType>::create(0.0)){};

        T_FloatType delta_t;
        vec3<T_FloatType> cell;

        struct Particle
        {
            uint32_t num_per_cell;
            T_FloatType density;
        };
        Particle particle;

        void convertToPic(Simulation& sim);
    };
} // namespace picongpu::simulation::cfg

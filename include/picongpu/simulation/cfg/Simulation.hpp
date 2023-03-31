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

#include "picongpu/simulation/cfg/Base.hpp"
#include "picongpu/simulation/cfg/Conversion.hpp"
#include "picongpu/simulation/cfg/Param.hpp"
#include "picongpu/simulation/cfg/PhysicalConstant.hpp"
#include "picongpu/simulation/cfg/Unit.hpp"

namespace picongpu::simulation::cfg
{
    struct Simulation
    {
        template<typename T_FloatType>
        struct SimulationData : Param<T_FloatType>
        {
            Unit<T_FloatType> unit;
            PhysicalConstant<T_FloatType> physicalConstant;
            Conversion<T_FloatType> conversion;
            Base<T_FloatType> base;
        };
        using Si = SimulationData<float_64>;
        Si si;
        using Pic = SimulationData<float_X>;
        Pic pic;

        void updateSi();
        void convertToPic();
    };

} // namespace picongpu::simulation::cfg

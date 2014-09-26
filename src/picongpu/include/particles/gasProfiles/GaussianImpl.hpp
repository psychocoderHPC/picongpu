/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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

#include "simulation_defines.hpp"

namespace picongpu
{

namespace gasProfiles
{

template<typename T_ParamClass>
struct GaussianImpl : public T_ParamClass
{
    typedef T_ParamClass ParamClass;

    /** Calculate the gas density, divided by the maximum density GAS_DENSITY
     *
     * @param globalCellOffset global offset to total domain [in cells]
     * @param cellSize size of a cell [in PIConGPU length]
     * @return float_X between 0.0 and 1.0
     */
    template<typename T_CellSizeType>
    HDINLINE float_X operator()(const DataSpace<simDim>& globalCellOffset,
                                const T_CellSizeType& cellSize)
    {
        const float_X vacuum_y = float_64(ParamClass::VACUUM_CELLS_Y) / getUnitValue(unit_length_) * cellSize.y();
        const float_X gas_center_left = ParamClass::SI::GAS_CENTER_LEFT / getUnitValue(unit_length_);
        const float_X gas_center_right = ParamClass::SI::GAS_CENTER_RIGHT / getUnitValue(unit_length_);
        const float_X gas_sigma_left = ParamClass::SI::GAS_SIGMA_LEFT / getUnitValue(unit_length_);
        const float_X gas_sigma_right = ParamClass::SI::GAS_SIGMA_RIGHT / getUnitValue(unit_length_);

        PMacc::math::Vector<float_X, simDim> globalParticlePos(
                                                               precisionCast<float_X>(globalCellOffset) * cellSize);

        if (globalParticlePos.y() * cellSize.y() < vacuum_y) return float_X(0.0);

        float_X exponent = float_X(0.0);
        if (globalParticlePos.y() < gas_center_left)
            exponent = math::abs((globalParticlePos.y() - gas_center_left) / gas_sigma_left);
        else if (globalParticlePos.y() > gas_center_right)
            exponent = math::abs((globalParticlePos.y() - gas_center_right) / gas_sigma_right);

        const float_X gas_power = ParamClass::GAS_POWER;
        const float_X density = math::exp(float_X(ParamClass::GAS_FACTOR) * math::pow(exponent, gas_power));
        return density;
    }

};
} //namespace gasProfiles
} //namespace picongpu

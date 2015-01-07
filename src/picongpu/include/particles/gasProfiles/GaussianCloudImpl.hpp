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
#include "simulationControl/MovingWindow.hpp"

namespace picongpu
{

namespace gasProfiles
{

template<typename T_ParamClass>
struct GaussianCloudImpl : public T_ParamClass
{
    typedef T_ParamClass ParamClass;

    template<typename T_SpeciesName>
    struct apply
    {
        typedef GaussianCloudImpl<ParamClass> type;
    };

    HINLINE GaussianCloudImpl(uint32_t currentStep)
    {
        const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        DataSpace<simDim> localCells = subGrid.getLocalDomain().size;
        gpuCellOffset = subGrid.getLocalDomain().offset;
        gpuCellOffset.y() += numSlides * localCells.y();
    }

    /** Calculate the gas density, divided by the maximum density GAS_DENSITY
     *
     * @param globalCellOffset global offset to total domain [in cells]
     * @param cellSize size of a cell [in PIConGPU length]
     * @return float_X between 0.0 and 1.0
     */
    HDINLINE float_X operator()(const DataSpace<simDim>& localCellIdx)
    {
        const float_64 unit_length = UNIT_LENGTH;
        const float_X vacuum_y = float_64(ParamClass::vacuum_y_cells) * cellSize.y();
        const floatD_X center = precisionCast<float>(ParamClass::SI().center / unit_length);
        const floatD_X sigma = precisionCast<float>(ParamClass::SI().sigma / unit_length);


        const floatD_X globalCellPos(
                                     precisionCast<float_X>(gpuCellOffset + localCellIdx) *
                                     cellSize
                                     );

        if (globalCellPos.y() < vacuum_y) return float_X(0.0);

        floatD_X exponent = float_X(math::abs((globalCellPos - center) / sigma));


        float_X density = 1;
        const float_X power = ParamClass::power;
        for (uint32_t d = 0; d < simDim; ++d)
            density *= math::exp(ParamClass::factor * math::pow(exponent[d], power));

        return density;
    }

protected:

    DataSpace<simDim> gpuCellOffset;

};
} //namespace gasProfiles
} //namespace picongpu

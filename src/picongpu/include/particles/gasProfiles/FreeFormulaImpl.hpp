/**
 * Copyright 2015 Rene Widera
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
struct FreeFormulaImpl : public T_ParamClass
{
    typedef T_ParamClass ParamClass;

    template<typename T_SpeciesName>
    struct apply
    {
        typedef FreeFormulaImpl<ParamClass> type;
    };

    HINLINE FreeFormulaImpl(uint32_t currentStep)
    {
        const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        DataSpace<simDim> localCells = subGrid.getLocalDomain().size;
        gpuCellOffset = subGrid.getLocalDomain().offset;
        gpuCellOffset.y() += numSlides * localCells.y();
    }

    /** Calculate the gas density
     *
     * @param totalCellOffset total offset including all slides [in cells]
     */
    HDINLINE float_X operator()(const DataSpace<simDim>& localCellIdx)
    {
        const float_64 unit_length = UNIT_LENGTH;

        float_X density = ParamClass::operator()( gpuCellOffset + localCellIdx, unit_length );

        return density;
    }

protected:

    DataSpace<simDim> gpuCellOffset;

};
} //namespace gasProfiles
} //namespace picongpu

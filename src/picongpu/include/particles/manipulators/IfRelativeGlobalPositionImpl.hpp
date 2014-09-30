/**
 * Copyright 2014 Rene Widera
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
namespace particles
{
namespace manipulators
{

template<typename T_ParamClass, typename T_Functor, typename T_SpeciesName = bmpl::_1>
struct IfRelativeGlobalPositionImpl : protected T_Functor
{
    typedef T_ParamClass ParamClass;
    typedef T_SpeciesName SpeciesName;
    typedef T_Functor Functor;
    typedef typename SpeciesName::type SpeciesType;
    typedef typename SpeciesType::FrameType FrameType;

    HINLINE IfRelativeGlobalPositionImpl(uint32_t currentStep)
    {
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        globalDomainSize = subGrid.getGlobalDomain().size;
        localDomainOffset = subGrid.getLocalDomain().offset;
    }

    template<typename T_Particle>
    DINLINE void operator()(const DataSpace<simDim>& localCellIdx, T_Particle& particle)
    {
        DataSpace<simDim> myCellPosition = localCellIdx + localDomainOffset;

        uint32_t relativePosition = myCellPosition[ParamClass::dimension] /
            globalDomainSize[ParamClass::dimension];

        if (ParamClass::lowerBound <= relativePosition &&
            ParamClass::upperBound > relativePosition)
        {
            Functor::operator()(localCellIdx, particle);
        }
    }

protected:

    DataSpace<simDim> localDomainOffset;
    DataSpace<simDim> globalDomainSize;
};

} //namespace manipulators
} //namespace particles
} //namespace picongpu

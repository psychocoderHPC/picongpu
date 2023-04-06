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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/param/dimension.param"
#include "picongpu/param/precision.param"
#include "pmacc/mappings/simulation/SubGrid.hpp"

#include <pmacc/mappings/kernel/MappingDescription.hpp>

#include <cupla.hpp>

namespace picongpu::simulation::cfg
{
    inline MappingDesc getDomainDescription()
    {
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        GridLayout<simDim> layout(subGrid.getLocalDomain().size, GuardSize::toRT() * SuperCellSize::toRT());
        return {layout.getDataSpace(), DataSpace<simDim>(GuardSize::toRT())};
    }
} // namespace picongpu::simulation::cfg
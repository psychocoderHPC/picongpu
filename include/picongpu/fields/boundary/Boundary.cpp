/* Copyright 2013-2023 Axel Huebl, Rene Widera, Sergei Bastrakov
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

#include "picongpu/fields/boundary/Boundary.hpp"

#include "picongpu/fields/boundary/impl/Thickness.hpp"
#include "picongpu/param/precision.param"
#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>

#include <pmacc/verify.hpp>

#include <cstdint>
#include <string>


namespace picongpu::fields::boundary
{
    impl::Thickness Boundary::getGlobalThickness() const
    {
        impl::Thickness thickness;
        auto numCells = m_boundary->getNumCells();
        for(uint32_t axis = 0u; axis < 3u; axis++)
            for(uint32_t direction = 0u; direction < 2u; direction++)
                thickness(axis, direction) = numCells[axis][direction];
        const pmacc::DataSpace<DIM3> isPeriodicBoundary
            = pmacc::Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();
        for(uint32_t axis = 0u; axis < 3u; axis++)
            if(isPeriodicBoundary[axis])
            {
                thickness(axis, 0) = 0u;
                thickness(axis, 1) = 0u;
            }
        return thickness;
    }
} // namespace picongpu::fields::boundary

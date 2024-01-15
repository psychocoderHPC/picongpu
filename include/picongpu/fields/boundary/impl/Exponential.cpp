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

#include "picongpu/fields/boundary/impl/Exponential.kernel"

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/boundary/Exponential.hpp"
#include "picongpu/param/dimension.param"
#include "picongpu/param/precision.param"
#include "picongpu/simulation/cfg/domain.hpp"
#include "picongpu/simulation/control/MovingWindow.hpp"

#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/memory/dataTypes/Mask.hpp>

#include <cstdint>
#include <string>


namespace picongpu::fields::boundary::impl
{
    inline Exponential::Exponential(boundary::Exponential const& base) : m_numCells(base.m_numCells)
    {
        m_strength
            = {precisionCast<float_X>(base.m_strength.x()),
               precisionCast<float_X>(base.m_strength.y()),
               precisionCast<float_X>(base.m_strength.z())};
    }

    template<class BoxedMemory>
    inline void Exponential::run(float_X currentStep, BoxedMemory deviceBox)
    {
        for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
        {
            /* only call for planes: left right top bottom back front*/
            if(FRONT % i == 0 && !(Environment<simDim>::get().GridController().getCommunicationMask().isSet(i)))
            {
                uint32_t direction = 0; /*set direction to X (default)*/
                if(i >= BOTTOM && i <= TOP)
                    direction = 1; /*set direction to Y*/
                if(i >= BACK)
                    direction = 2; /*set direction to Z*/

                /* exchange mod 2 to find positive or negative direction
                 * positive direction = 1
                 * negative direction = 0
                 */
                uint32_t pos_or_neg = i % 2;

                uint32_t thickness = m_numCells[direction][pos_or_neg];
                float_X absorberStrength = m_strength[direction][pos_or_neg];

                if(thickness == 0)
                    continue; /*if the absorber has no thickness we check the next side*/


                /* if sliding window is active we disable absorber on bottom side*/
                if(MovingWindow::getInstance().isSlidingWindowActive(static_cast<uint32_t>(currentStep))
                   && i == BOTTOM)
                    continue;

                auto domDesc = simulation::cfg::getDomainDescription();
                ExchangeMapping<GUARD, MappingDesc> mapper(domDesc, i);

                auto workerCfg = pmacc::lockstep::makeWorkerCfg(SuperCellSize{});
                PMACC_LOCKSTEP_KERNEL(KernelAbsorbBorder{}, workerCfg)
                (mapper.getGridDim())(deviceBox, thickness, absorberStrength, mapper);
            }
        }
    }

    template void Exponential::run(float_X, typename FieldB::DataBoxType);
    //template void Exponential::run(float_X, typename FieldE::DataBoxType);
} // namespace picongpu::fields::boundary::impl

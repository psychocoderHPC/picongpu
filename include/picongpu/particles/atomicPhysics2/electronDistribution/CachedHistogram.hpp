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

#include <pmacc/memory/Array.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::kernel
{
    template<uint32_t T_size>
    struct CachedHistogram
    {
        pmacc::memory::Array<float_X, T_size> energy;
        pmacc::memory::Array<float_X, T_size> collisionalRate;

        static constexpr uint32_t size = T_size;

        constexpr uint32_t numBins() const
        {
            return size;
        }

        /** Fill histogram
         *
         * @attention This method is synchronizing the worker before returning the handle.
         *
         * @tparam T_Worker
         * @tparam T_Histogram
         * @param worker
         * @param electronHistogram
         * @param volumeScalingFactor
         */
        template<typename T_Worker, typename T_Histogram, typename T_RateFunctor>
        HDINLINE void fill(
            T_Worker const& worker,
            T_Histogram const& electronHistogram,
            float_X const volumeScalingFactor,
            T_RateFunctor const& rateFunctor)
        {
            auto forEachElement = lockstep::makeForEach<T_size>(worker);
            forEachElement(
                [&](uint32_t const idx)
                {
                    float_X energyValue = electronHistogram.getBinEnergy(idx);
                    energy[idx] = energyValue;
                    // eV
                    float_X const binWith = electronHistogram.getBinWidth(idx);
                    // 1/(UNIT_LENGTH^3 * eV)
                    float_X const density = electronHistogram.getBinWeight0(idx) / volumeScalingFactor / binWith;
                    collisionalRate[idx] = rateFunctor(energyValue, binWith, density);
                });
            worker.sync();
        }
    };


} // namespace picongpu::particles::atomicPhysics2::kernel

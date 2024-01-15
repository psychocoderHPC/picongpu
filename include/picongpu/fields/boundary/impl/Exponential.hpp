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

#pragma once

#include "picongpu/param/precision.param"

#include <cstdint>


namespace picongpu::fields::boundary
{
    class Exponential;

    namespace impl
    {
        class Exponential
        {
            vec3<vec2<uint32_t>> m_numCells;
            vec3<vec2<float_X>> m_strength;

        public:
            Exponential(boundary::Exponential const&);

            /** Apply absorber to the given field
             *
             * @tparam BoxedMemory field box type
             *
             * @param currentStep current time iteration
             * @param deviceBox field box
             */
            template<class BoxedMemory>
            void run(float_X currentStep, BoxedMemory deviceBox);
        };
    } // namespace impl
} // namespace picongpu::fields::boundary

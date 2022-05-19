/* Copyright 2013-2022 Heiko Burau, Rene Widera, Axel Huebl, Sergei Bastrakov
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

#include <cstdint>


namespace picongpu
{
    namespace particles
    {
        namespace shapes
        {
            namespace detail
            {
                struct CIC
                {
                    /** Support of the assignment function in cells
                     *
                     * Specifies width of the area where the function can be non-zero.
                     * Is the same for all directions
                     */
                    static constexpr uint32_t support = 2;

                    /**
                     *
                     * @tparam T_size
                     * @param  x particle position relative to the assignment cell range [0.0;1.0)
                     * @return array with evaluated shape values
                     */
                    template<uint32_t T_size>
                    HDINLINE auto shapeArray(float_X const x) const
                    {
                        pmacc::memory::Array<float_X, T_size> shapeValues;
                        // grid point [0;1]
                        // note: math::abs(0 - x) == math::abs(x)
                        shapeValues[0] = math::abs(x);
                        shapeValues[1] = 1.0_X - x;
                        return shapeValues;
                    }
                };

            } // namespace detail

            /** Cloud-in-cell particle shape
             *
             * Cloud density form: piecewise constant
             * Assignment function: first order B-spline
             */
            struct CIC
            {
                //! Order of the assignment function spline
                static constexpr uint32_t assignmentFunctionOrder = detail::CIC::support - 1u;

                struct ChargeAssignment : public detail::CIC
                {
                    static constexpr int begin = 0;
                    static constexpr int end = 2;

                    HDINLINE float_X operator()(float_X const x) const
                    {
                        /*       -
                         *       |  1-|x|           if |x|<1
                         * W(x)=<|
                         *       |  0               otherwise
                         *       -
                         */
                        float_X const abs_x = math::abs(x);

                        bool const below_1 = abs_x < 1.0_X;
                        float_X const onSupport = 1.0_X - abs_x;

                        float_X result(0.0);
                        if(below_1)
                            result = onSupport;

                        return result;
                    }

                    // @param x particle position: range [0.0;2.0)
                    HDINLINE auto shapeArray(float_X const xx, bool const isOutOfRange) const
                    {
                        float_X x = isOutOfRange ? xx - 1.0_X : xx;

                        auto shapeValues = detail::CIC::shapeArray<support + 1>(x);

                        shapeValues[2] = isOutOfRange ? shapeValues[1] : 0.0_X;
                        shapeValues[1] = isOutOfRange ? shapeValues[0] : shapeValues[1];
                        shapeValues[0] = isOutOfRange ? 0.0_X : shapeValues[0];

                        return shapeValues;
                    }
                };

                struct ChargeAssignmentOnSupport : public detail::CIC
                {
                    static constexpr int begin = 0;
                    static constexpr int end = 1;

                    /** form factor of this particle shape.
                     * @param x has to be within [-support/2, support/2]
                     */
                    HDINLINE float_X operator()(float_X const x) const
                    {
                        /*
                         * W(x)=1-|x|
                         */
                        return 1.0_X - math::abs(x);
                    }

                    // @param x particle position: range [0.0;1.0)
                    HDINLINE auto shapeArray(float_X const x, bool const /*isOutOfRange*/) const
                    {
                        return detail::CIC::shapeArray<support>(x);
                    }
                };
            };

        } // namespace shapes
    } // namespace particles
} // namespace picongpu

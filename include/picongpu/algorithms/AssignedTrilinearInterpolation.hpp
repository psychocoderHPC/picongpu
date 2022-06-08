/* Copyright 2013-2022 Axel Huebl, Heiko Burau, Rene Widera
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

#include <pmacc/attribute/unroll.hpp>
#include <pmacc/result_of_Functor.hpp>
#include <pmacc/types.hpp>

#include <type_traits>


// forward declaration
namespace picongpu
{
    struct AssignedTrilinearInterpolation;
} // namespace picongpu

namespace pmacc
{
    namespace result_of
    {
        template<typename T_Cursor>
        struct Functor<picongpu::AssignedTrilinearInterpolation, T_Cursor>
        {
            using type = typename T_Cursor::ValueType;
        };

    } // namespace result_of
} // namespace pmacc

namespace picongpu
{
    struct AssignedTrilinearInterpolation
    {
        /** Does a trilinear field-to-point interpolation for
         * arbitrary assignment function and arbitrary field_value types.
         *
         * @tparam T_begin lower margin for interpolation
         * @tparam T_end upper margin for interpolation
         * @tparam T_AssignmentFunction type of shape functors
         *
         * @param cursor cursor pointing to the field
         * @param shapeFunctors Array with d shape functors, where d is the dimensionality of the field represented by
         *                      cursor. The shape functor must have the interface to call
         *                      `operator()(relative_grid_point)` and return the assignment value for the given grid
         *                      point.
         * @return sum over: field_value * assignment
         *
         * interpolate on grid points in range [T_begin;T_end]
         *
         * @{
         */
        template<int T_begin, int T_end, typename T_Cursor, typename T_AssignmentFunction>
        HDINLINE static auto interpolate(const T_Cursor& cursor, const T_AssignmentFunction& shapeFunctors)
        {
            [[maybe_unused]] constexpr auto iterations = T_end - T_begin + 1;

            /* The implementation assumes that x is the fastest moving index to iterate over contiguous memory
             * e.g. a row, to optimize memory fetch operations.
             */
            auto result_z = float4(0.0,0.0,0.0,0.0);
            PMACC_UNROLL(iterations)
            for(int z = T_begin; z <= T_end; ++z)
            {
                auto result_y = float4(0.0,0.0,0.0,0.0);
                PMACC_UNROLL(iterations)
                for(int y = T_begin; y <= T_end; ++y)
                {
                    auto result_x = float4(0.0,0.0,0.0,0.0);

                    float4 c;
                    float4 s;
                    PMACC_UNROLL(iterations)
                    for(int x = T_begin; x <= T_end; ++x)
                    {
                        /* a form factor is the "amount of particle" that is affected by this cell
                         * so we have to sum over: cell_value * form_factor
                         */
                        PMACC_UNROLL(3)
                        for(int d = 0; d < 3; ++d)
                        {
                            *((&c.x) + d) = *cursor[d](x, y, z);
                            *((&s.x) + d) = shapeFunctors[d][0](x);
                        }
                        result_x += c * s;
                    }

                    float4 s_y;
                    PMACC_UNROLL(3)
                    for(int d = 0; d < 3; ++d)
                        *((&s_y.x) + d) = shapeFunctors[d][1](y);

                    result_y += result_x * s_y;
                }
                float4 s_z;
                PMACC_UNROLL(3)
                for(int d = 0; d < 3; ++d)
                    *((&s_z.x) + d) = shapeFunctors[d][2](z);
                result_z += result_y * s_z;
            }
            float3_X tmp;
            for(int d = 0; d < 3; ++d)
                tmp[d] = *((&result_z.x) + d);
            return tmp;
        }

        /** Implementation for 2D position*/
        template<int T_begin, int T_end, class T_Cursor, class T_AssignmentFunction>
        HDINLINE static auto interpolate(
            T_Cursor const& cursor,
            const pmacc::memory::Array<T_AssignmentFunction, 2>& shapeFunctors)
        {
            [[maybe_unused]] constexpr int iterations = T_end - T_begin + 1;

            using type = decltype(*cursor(0, 0) * shapeFunctors[0](0));
            /* The implementation assumes that x is the fastest moving index to iterate over contiguous memory
             * e.g. a row, to optimize memory fetch operations.
             */
            auto result_y = type(0.0);
            PMACC_UNROLL(iterations)
            for(int y = T_begin; y <= T_end; ++y)
            {
                auto result_x = type(0.0);
                PMACC_UNROLL(iterations)
                for(int x = T_begin; x <= T_end; ++x)
                    // a form factor is the "amount of particle" that is affected by this cell
                    // so we have to sum over: cell_value * form_factor
                    result_x += *cursor(x, y) * shapeFunctors[0](x);

                result_y += result_x * shapeFunctors[1](y);
            }
            return result_y;
        }

        static auto getStringProperties() -> pmacc::traits::StringProperty
        {
            pmacc::traits::StringProperty propList("name", "uniform");
            return propList;
        }
    };

    //! @}

} // namespace picongpu

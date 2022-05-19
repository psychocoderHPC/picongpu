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
        /** Does a 3D trilinear field-to-point interpolation for
         * arbitrary assignment function and arbitrary field_value types.
         *
         * @tparam T_begin lower margin for interpolation
         * @tparam T_end upper margin for interpolation
         * @tparam T_AssignmentFunction function for assignment
         *
         * @param cursor cursor pointing to the field
         * @param pos position of the interpolation point
         * @return sum over: field_value * assignment
         *
         * interpolate on grid points in range [T_begin;T_end]
         */
        template<int T_begin, int T_end, typename T_Cursor, typename T_AssignmentFunction>
        HDINLINE static auto interpolate(
            const T_Cursor& cursor,
            const pmacc::memory::Array<T_AssignmentFunction, 3>& shapeFunctor)
        {
            [[maybe_unused]] constexpr auto iterations = T_end - T_begin + 1;

            auto result_z = float_X(0.0);
            PMACC_UNROLL(iterations)
            for(int z = T_begin; z <= T_end; ++z)
            {
                auto result_y = float_X(0.0);
                PMACC_UNROLL(iterations)
                for(int y = T_begin; y <= T_end; ++y)
                {
                    auto result_x = float_X(0.0);
                    PMACC_UNROLL(iterations)
                    for(int x = T_begin; x <= T_end; ++x)
                        /* a form factor is the "amount of particle" that is affected by this cell
                         * so we have to sum over: cell_value * form_factor
                         */
                        result_x += *cursor(x, y, z) * shapeFunctor[0](x);

                    result_y += result_x * shapeFunctor[1](y);
                }

                result_z += result_y * shapeFunctor[2](z);
            }
            return result_z;
        }

        /** Implementation for 2D position*/
        template<int T_begin, int T_end, class T_Cursor, class T_AssignmentFunction>
        HDINLINE static auto interpolate(
            T_Cursor const& cursor,
            const pmacc::memory::Array<T_AssignmentFunction, 2>& shapeFunctor)
        {
            constexpr int iterations = T_end - T_begin + 1;

            auto result_y = float_X(0.0);
            PMACC_UNROLL(iterations)
            for(int y = T_begin; y <= T_end; ++y)
            {
                auto result_x = float_X(0.0);
                PMACC_UNROLL(iterations)
                for(int x = T_begin; x <= T_end; ++x)
                    // a form factor is the "amount of particle" that is affected by this cell
                    // so we have to sum over: cell_value * form_factor
                    result_x += *cursor(x, y) * shapeFunctor[0](x);

                result_y += result_x * shapeFunctor[1](y);
            }
            return result_y;
        }

        static auto getStringProperties() -> pmacc::traits::StringProperty
        {
            pmacc::traits::StringProperty propList("name", "uniform");
            return propList;
        }
    };

} // namespace picongpu
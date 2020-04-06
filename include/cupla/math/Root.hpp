/* Copyright 2020 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */


#pragma once

#include "cupla/math/Common.hpp"
#include "cupla/types.hpp"

namespace cupla
{
inline namespace CUPLA_ACCELERATOR_NAMESPACE
{
inline namespace device
{
inline namespace math
{

    //! Computes the square root.
    CUPLA_UNARY_MATH_FN( sqrt, alpaka::math::ConceptMathSqrt, Sqrt )

    //! Computes the inverse square root.
    CUPLA_UNARY_MATH_FN( rsqrt, alpaka::math::ConceptMathRsqrt, Rsqrt )

    //! Computes the cubic root.
    CUPLA_UNARY_MATH_FN( cbrt, alpaka::math::ConceptMathCbrt, Cbrt )

} // namespace math
} // namespace device
} // namespace CUPLA_ACCELERATOR_NAMESPACE
} // namespace cupla

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

#include "cupla/types.hpp"

#include <alpaka/core/Concepts.hpp>

#include <type_traits>

namespace cupla
{
inline namespace CUPLA_ACCELERATOR_NAMESPACE
{
inline namespace device
{
inline namespace math
{
namespace detail
{
#if defined(__CUDA_ARCH__) || ( defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__== 1 && defined(__HIP__) )
    #define CUPLA_HIP_DEVICE_PATH 1
#else
    #define CUPLA_HIP_DEVICE_PATH 0
#endif

    /** Get the concept implementation of the current accelerator
     *
     * @tparam T_Concept alpaka concept
     * @return implementation of the concept
     */
    ALPAKA_NO_HOST_ACC_WARNING
    template< typename T_AccOrMathImpl, typename T_Concept >
    ALPAKA_FN_HOST_ACC  auto getConcept()
    {
        using AccMathConcept = alpaka::concepts::ImplementationBase<
            T_Concept,
            T_AccOrMathImpl
        >;
#if CUPLA_HIP_DEVICE_PATH == 1
        using AccThreadSeqMathConcept = alpaka::concepts::ImplementationBase<
            T_Concept,
            AccThreadSeq
        >;

        // cupla Acc and AccThreadSeq should use the same math concept implementation
        static_assert(
            std::is_same<
                AccMathConcept,
                AccThreadSeqMathConcept
            >::value,
            "The math concept implementation for the type 'Acc' and 'AccThreadSeq' must be equal"
        );
#endif
        return AccMathConcept{};
    }

} // namespace detail

#define CUPLA_UNARY_MATH_FN_DETAIL(accOrMathImpl, functionName, alpakaMathConcept, alpakaMathTrait)  \
    /**                                                                        \
     * @tparam T_Type argument type                                            \
     * @param arg input argument                                               \
     */                                                                        \
    ALPAKA_NO_HOST_ACC_WARNING                                                 \
    template< typename T_Type >                                                \
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto functionName(                     \
        T_Type const & arg                                                     \
    )                                                                          \
    /* return type is required because nvcc can not detect the return type     \
     * for device functions.                                                   \
     */                                                                        \
    ->  decltype(                                                              \
        alpaka::math::traits::alpakaMathTrait<                                 \
            alpaka::concepts::ImplementationBase<                              \
                alpakaMathConcept,                                             \
                accOrMathImpl                                                  \
            >,                                                                 \
            T_Type                                                             \
        >::functionName(                                                       \
            detail::getConcept< accOrMathImpl, alpakaMathConcept >(),          \
            arg                                                                \
        )                                                                      \
    )                                                                          \
    {                                                                          \
        return alpaka::math::traits::alpakaMathTrait<                          \
            alpaka::concepts::ImplementationBase<                              \
                alpakaMathConcept,                                             \
                accOrMathImpl                                                  \
            >,                                                                 \
            T_Type                                                             \
        >::functionName(                                                       \
            detail::getConcept< accOrMathImpl, alpakaMathConcept >(),          \
            arg                                                                \
        );                                                                     \
    }

/* Using the free alpaka functions `alpaka::math::*` will result into `__host__ __device__`
 * errors, therefore the alpaka math trait must be used.
 */
#define CUPLA_BINARY_MATH_FN_DETAIL(accOrMathImpl, functionName, alpakaMathConcept, alpakaMathTrait) \
    /**                                                                        \
     * @tparam T_Type argument type                                            \
     * @param arg1 first input argument                                        \
     * @param arg2 second input argument                                       \
     */                                                                        \
    template<                                                                  \
        typename T_Type1,                                                      \
        typename T_Type2                                                       \
    >                                                                          \
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto functionName(                     \
        T_Type1 const & arg1,                                                  \
        T_Type2 const & arg2                                                   \
    )                                                                          \
    /* return type is required because nvcc can not detect the return type     \
     * for device functions.                                                   \
     */                                                                        \
    ->  decltype(                                                              \
        alpaka::math::traits::alpakaMathTrait<                                 \
            alpaka::concepts::ImplementationBase<                              \
                alpakaMathConcept,                                             \
                accOrMathImpl                                                  \
            >,                                                                 \
            T_Type1,                                                           \
            T_Type2                                                            \
        >::functionName(                                                       \
            detail::getConcept< accOrMathImpl, alpakaMathConcept >(),          \
            arg1,                                                              \
            arg2                                                               \
        )                                                                      \
    )                                                                          \
    {                                                                          \
        return alpaka::math::traits::alpakaMathTrait<                          \
            alpaka::concepts::ImplementationBase<                              \
                alpakaMathConcept,                                             \
                accOrMathImpl                                                  \
            >,                                                                 \
            T_Type1,                                                           \
            T_Type2                                                            \
        >::functionName(                                                       \
            detail::getConcept< accOrMathImpl, alpakaMathConcept >(),          \
            arg1,                                                              \
            arg2                                                               \
        );                                                                     \
    }

#if CUPLA_HIP_DEVICE_PATH == 0
    #define CUPLA_UNARY_MATH_FN(functionName, alpakaMathConcept, alpakaMathTrait) CUPLA_UNARY_MATH_FN_DETAIL(alpaka::math::MathStdLib, functionName, alpakaMathConcept, alpakaMathTrait)
    #define CUPLA_BINARY_MATH_FN(functionName, alpakaMathConcept, alpakaMathTrait) CUPLA_BINARY_MATH_FN_DETAIL(alpaka::math::MathStdLib, functionName, alpakaMathConcept, alpakaMathTrait)
#else
    #define CUPLA_UNARY_MATH_FN(functionName, alpakaMathConcept, alpakaMathTrait) CUPLA_UNARY_MATH_FN_DETAIL(Acc, functionName, alpakaMathConcept, alpakaMathTrait)
    #define CUPLA_BINARY_MATH_FN(functionName, alpakaMathConcept, alpakaMathTrait) CUPLA_BINARY_MATH_FN_DETAIL(Acc, functionName, alpakaMathConcept, alpakaMathTrait)
#endif

} // namespace math
} // namespace device
} // namespace CUPLA_ACCELERATOR_NAMESPACE
} // namespace cupla

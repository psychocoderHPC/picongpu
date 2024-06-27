/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2024 Helmholtz-Zentrum Dresden - Rossendorf,
                 CERN

  Author(s):  Julian Johannes Lenz

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/


#include <alpaka/core/Common.hpp>
#include <cstddef>
#include <cstdint>

template<typename TAcc, typename TFunctor, typename... TArgs>
ALPAKA_FN_ACC [[nodiscard]] inline auto internalWrappingLoop(
    TAcc const& acc,
    size_t const startIndex,
    size_t const endIndex,
    auto failureValue,
    TFunctor func,
    TArgs... args)
{
    auto result = failureValue;
    for(uint32_t i = startIndex; i < endIndex && result == failureValue; ++i)
    {
        result = func(acc, i, args...);
    }
    return result;
}

template<typename TAcc, typename TFunctor, typename... TArgs>
ALPAKA_FN_ACC [[nodiscard]] inline auto wrappingLoop(
    TAcc const& acc,
    size_t const startIndex,
    size_t const size,
    auto failureValue,
    TFunctor func,
    TArgs... args)
{
    auto result = internalWrappingLoop(acc, startIndex, size, failureValue, func, args...);
    if(result == failureValue)
    {
        result = internalWrappingLoop(acc, 0U, startIndex, failureValue, func, args...);
    }
    return result;
}

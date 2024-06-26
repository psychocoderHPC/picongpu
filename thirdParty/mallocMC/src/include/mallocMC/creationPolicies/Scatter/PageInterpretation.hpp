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

#pragma once

#include "mallocMC/auxiliary.hpp"
#include "mallocMC/creationPolicies/Scatter/BitField.hpp"
#include "mallocMC/creationPolicies/Scatter/DataPage.hpp"

#include <cstdint>
#include <cstring>
#include <unistd.h>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    template<size_t T_pageSize>
    struct PageInterpretation
    {
    private:
        DataPage<T_pageSize>& _data;
        uint32_t const _chunkSize;

    public:
        // this is needed to instantiate this in-place in an std::optional
        ALPAKA_FN_ACC PageInterpretation(DataPage<T_pageSize>& data, uint32_t chunkSize)
            : _data(data)
            , _chunkSize(chunkSize)
        {
        }

        ALPAKA_FN_ACC static auto bitFieldStart(DataPage<T_pageSize>& data, uint32_t const chunkSize) -> BitMask*
        {
            return PageInterpretation<T_pageSize>(data, chunkSize).bitFieldStart();
        }

        ALPAKA_FN_ACC [[nodiscard]] constexpr static auto numChunks(uint32_t const chunkSize) -> uint32_t
        {
            return BitMaskSize * T_pageSize / (static_cast<size_t>(BitMaskSize * chunkSize) + sizeof(BitMask));
        }

        ALPAKA_FN_ACC [[nodiscard]] auto numChunks() const -> uint32_t
        {
            return numChunks(_chunkSize);
        }

        ALPAKA_FN_ACC [[nodiscard]] auto operator[](size_t index) const -> void*
        {
            return reinterpret_cast<void*>(&_data.data[index * _chunkSize]);
        }

        template<typename TAcc>
        ALPAKA_FN_ACC auto create(TAcc const& acc) -> void*
        {
            auto field = bitField();
            auto const index = firstFreeBit(acc, field, numChunks());
            return (index < noFreeBitFound(field)) ? this->operator[](index) : nullptr;
        }

        template<typename TAcc>
        ALPAKA_FN_ACC auto destroy(TAcc const& acc, void* pointer) -> void
        {
            if(_chunkSize == 0)
            {
#ifndef NDEBUG
                throw std::runtime_error{
                    "Attempted to destroy a pointer with chunkSize==0. Likely this page was recently "
                    "(and potentially pre-maturely) freed."};
#endif // NDEBUG
                return;
            }
            auto chunkIndex = chunkNumberOf(pointer);
#ifndef NDEBUG
            if(not isValid(acc, chunkIndex))
            {
                throw std::runtime_error{"Attempted to destroy an invalid pointer! Either the pointer does not point "
                                         "to a valid chunk or it is not marked as allocated."};
            }
#endif // NDEBUG
            bitField().unset(acc, chunkIndex);
        }

        ALPAKA_FN_ACC auto cleanup() -> void
        {
            // This method is not thread-safe by itself. But it is supposed to be called after acquiring a "lock" in
            // the form of setting the filling level, so that's fine.
            memset(&_data.data[T_pageSize - maxBitFieldSize()], 0U, maxBitFieldSize());
        }

        template<typename TAcc>
        ALPAKA_FN_ACC auto isValid(TAcc const& acc, void* pointer) -> bool
        {
            // This function is neither thread-safe nor particularly performant. It is supposed to be used in tests and
            // debug mode.
            return isValid(acc, chunkNumberOf(pointer));
        }

    private:
        template<typename TAcc>
        ALPAKA_FN_ACC auto isValid(TAcc const& acc, ssize_t const chunkIndex) -> bool
        {
            return chunkIndex >= 0 and chunkIndex < numChunks() and isAllocated(acc, chunkIndex);
        }

    public:
        template<typename TAcc>
        ALPAKA_FN_ACC auto isAllocated(TAcc const& acc, uint32_t const chunkIndex) -> bool
        {
            return bitField().get(acc, chunkIndex);
        }

        ALPAKA_FN_ACC [[nodiscard]] auto bitField() const -> BitFieldFlat
        {
            return BitFieldFlat{{bitFieldStart(), ceilingDivision(numChunks(), BitMaskSize)}};
        }

        ALPAKA_FN_ACC [[nodiscard]] auto bitFieldStart() const -> BitMask*
        {
            return reinterpret_cast<BitMask*>(&_data.data[T_pageSize - bitFieldSize()]);
        }

        ALPAKA_FN_ACC [[nodiscard]] auto bitFieldSize() const -> uint32_t
        {
            return bitFieldSize(_chunkSize);
        }

        ALPAKA_FN_ACC [[nodiscard]] static auto bitFieldSize(uint32_t const chunkSize) -> uint32_t
        {
            return sizeof(BitMask) * ceilingDivision(numChunks(chunkSize), BitMaskSize);
        }

        ALPAKA_FN_ACC [[nodiscard]] static auto maxBitFieldSize() -> uint32_t
        {
            return PageInterpretation<T_pageSize>::bitFieldSize(1U);
        }

        ALPAKA_FN_ACC [[nodiscard]] auto chunkNumberOf(void* pointer) -> ssize_t
        {
            return indexOf(pointer, &_data, _chunkSize);
        }

        // these are supposed to be temporary objects, don't start messing around with them:
        PageInterpretation(PageInterpretation const&) = delete;
        PageInterpretation(PageInterpretation&&) = delete;
        auto operator=(PageInterpretation const&) -> PageInterpretation& = delete;
        auto operator=(PageInterpretation&&) -> PageInterpretation& = delete;
        ~PageInterpretation() = default;
    };
} // namespace mallocMC::CreationPolicies::ScatterAlloc

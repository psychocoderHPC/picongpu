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

#include <alpaka/core/Common.hpp>
#include <alpaka/intrinsic/Traits.hpp>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    constexpr const uint32_t BitMaskSize = 32U;
    template<uint32_t size = BitMaskSize, typename = std::enable_if_t<BitMaskSize == 32U>> // NOLINT(*magic-number*)
    using BitMaskStorageType = uint32_t;
    ALPAKA_STATIC_ACC_MEM_CONSTANT constexpr const BitMaskStorageType<BitMaskSize> allOnes
        = std::numeric_limits<BitMaskStorageType<BitMaskSize>>::max();

    ALPAKA_FN_ACC inline auto singleBit(BitMaskStorageType<> const index) -> BitMaskStorageType<>
    {
        return 1U << index;
    }

    template<typename TAcc>
    ALPAKA_FN_ACC inline auto allOnesUpTo(BitMaskStorageType<> const index) -> BitMaskStorageType<>
    {
        if(index == 0U)
            return 0U;
        if(index == BitMaskSize)
            return allOnes<TAcc>;
        // A global constexpr variable cannot be referenced (because it doesn't necessarily exist in memory but could
        // be inlined everywhere by the compiler). Most compilers don't really care but nvcc apparently does, so we get
        // a variable in local scope to reference BitMaskSize.
        constexpr auto referencableBitMaskSize = BitMaskSize;
        return allOnes<TAcc> >> (referencableBitMaskSize - std::min(std::max(index, 0U), referencableBitMaskSize));
    }

    struct BitMask
    {
        // Convention: We start counting from the right, i.e., if mask[0] == 1 and all others are 0, then mask = 0...01
        BitMaskStorageType<BitMaskSize> mask{};

        template<typename TAcc>
        ALPAKA_FN_ACC auto operator()(TAcc const& acc, auto const index)
        {
            return (atomicLoad(acc, mask) & singleBit(index)) != 0U;
        }

        template<typename TAcc>
        ALPAKA_FN_ACC auto set(TAcc const& acc)
        {
            atomicOr(acc, mask, allOnes<TAcc>);
        }

        template<typename TAcc>
        ALPAKA_FN_ACC auto set(TAcc const& acc, auto const index)
        {
            return atomicOr(acc, mask, singleBit(index));
        }

        template<typename TAcc>
        ALPAKA_FN_ACC auto unset(TAcc const& acc, auto const index)
        {
            return atomicAnd(acc, mask, allOnes<TAcc> ^ singleBit(index));
        }

        template<typename TAcc>
        ALPAKA_FN_ACC auto flip(TAcc const& acc)
        {
            return atomicXor(acc, mask, allOnes<TAcc>);
        }

        template<typename TAcc>
        ALPAKA_FN_ACC auto flip(TAcc const& acc, auto const index)
        {
            return atomicXor(acc, mask, singleBit(index));
        }

        ALPAKA_FN_ACC auto operator==(auto const other) const
        {
            return (mask == other);
        }

        // clang-format off
         ALPAKA_FN_ACC auto operator<=> (BitMask const other) const
        // clang-format on
        {
            return (mask - other.mask);
        }

        ALPAKA_FN_ACC [[nodiscard]] auto none() const
        {
            return mask == 0U;
        }

        template<typename TAcc>
        ALPAKA_FN_ACC [[nodiscard]] auto all(TAcc const& acc) const
        {
            return atomicAnd(acc, mask, allOnes<TAcc>);
        }
    };

    struct BitFieldFlat
    {
        std::span<BitMask> data;

        template<typename TAcc>
        ALPAKA_FN_ACC [[nodiscard]] auto get(TAcc const& acc, uint32_t index) const -> bool
        {
            return data[index / BitMaskSize](acc, index % BitMaskSize);
        }

        ALPAKA_FN_ACC [[nodiscard]] auto operator[](uint32_t index) const -> BitMask&
        {
            return data[index];
        }

        template<typename TAcc>
        ALPAKA_FN_ACC void set(TAcc const& acc, uint32_t const index) const
        {
            data[index / BitMaskSize].set(acc, index % BitMaskSize);
        }

        template<typename TAcc>
        ALPAKA_FN_ACC void unset(TAcc const& acc, uint32_t const index) const
        {
            data[index / BitMaskSize].unset(acc, index % BitMaskSize);
        }

        ALPAKA_FN_ACC [[nodiscard]] auto begin() const
        {
            return std::begin(data);
        }

        ALPAKA_FN_ACC [[nodiscard]] auto end() const
        {
            return std::end(data);
        }

        ALPAKA_FN_ACC [[nodiscard]] auto numMasks() const
        {
            return data.size();
        }

        ALPAKA_FN_ACC [[nodiscard]] auto numBits() const
        {
            return numMasks() * BitMaskSize;
        }
    };

    ALPAKA_FN_ACC inline auto noFreeBitFound(BitMask const& /*unused*/) -> uint32_t
    {
        return BitMaskSize;
    }

    ALPAKA_FN_ACC inline auto noFreeBitFound(BitFieldFlat const& field) -> uint32_t
    {
        return field.numBits();
    }

    template<typename TAcc>
    ALPAKA_FN_ACC [[nodiscard]] inline auto firstFreeBitInBetween(
        TAcc const& acc,
        BitMask& mask,
        uint32_t const startIndex,
        uint32_t const endIndex) -> uint32_t
    {
        auto result = noFreeBitFound(mask);
        for(uint32_t i = startIndex; i >= startIndex and i < endIndex and result == noFreeBitFound(mask);)
        {
            auto oldMask = atomicOr(acc, mask.mask, singleBit(i)) | allOnesUpTo<TAcc>(startIndex);
            if((oldMask & singleBit(i)) == 0U)
            {
                result = i;
            }
            i = oldMask == allOnes<TAcc>
                ? noFreeBitFound(mask)
                : alpaka::ffs(acc, static_cast<std::make_signed_t<BitMaskStorageType<>>>(~oldMask)) - 1;
        }
        return result;
    }

    template<typename TAcc>
    ALPAKA_FN_ACC [[nodiscard]] inline auto firstFreeBit(TAcc const& acc, BitMask& mask, uint32_t const startIndex = 0)
        -> uint32_t
    {
        auto result = firstFreeBitInBetween(acc, mask, startIndex, BitMaskSize);
        if(result == noFreeBitFound(mask))
        {
            result = firstFreeBitInBetween(acc, mask, 0U, startIndex);
        }
        return result;
    }

    template<typename TAcc>
    ALPAKA_FN_ACC inline auto firstFreeBit(TAcc const& acc, BitFieldFlat& field, uint32_t numValidBits = 0) -> uint32_t
    {
        if(numValidBits == 0)
        {
            numValidBits = field.numBits();
        }
        for(uint32_t i = 0; i < field.numMasks(); ++i)
        {
            auto indexInMask = firstFreeBit(acc, field[i]);
            if(indexInMask < noFreeBitFound(BitMask{}))
            {
                uint32_t freeBitIndex = indexInMask + BitMaskSize * i;
                if(freeBitIndex < numValidBits)
                {
                    return freeBitIndex;
                }
                return noFreeBitFound(field);
            }
        }
        return noFreeBitFound(field);
    }

} // namespace mallocMC::CreationPolicies::ScatterAlloc

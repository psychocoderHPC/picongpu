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


#include "mallocMC/auxiliary.hpp"

#include <algorithm>
#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/acc/AccCpuThreads.hpp>
#include <alpaka/acc/Tag.hpp>
#include <alpaka/acc/TagAccIsEnabled.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/mem/alloc/Traits.hpp>
#include <alpaka/mem/buf/BufCpu.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/platform/PlatformCpu.hpp>
#include <alpaka/platform/Traits.hpp>
#include <alpaka/queue/Properties.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>
#include <mallocMC/creationPolicies/Scatter.hpp>
#include <span>
#include <tuple>
#include <type_traits>

using mallocMC::CreationPolicies::ScatterAlloc::AccessBlock;
using mallocMC::CreationPolicies::ScatterAlloc::BitMaskSize;

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;


constexpr uint32_t pageSize = 1024;
constexpr size_t numPages = 4;
// Page table entry size = sizeof(chunkSize) + sizeof(fillingLevel):
constexpr uint32_t pteSize = 4 + 4;
constexpr size_t blockSize = numPages * (pageSize + pteSize);

using MyAccessBlock = AccessBlock<blockSize, pageSize>;

// Fill all pages of the given access block with occupied chunks of the given size. This is useful to test the
// behaviour near full filling but also to have a deterministic page and chunk where an allocation must happen
// regardless of the underlying access optimisations etc.

struct FillWith
{
    template<typename TAcc, size_t T_blockSize, uint32_t T_pageSize>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        AccessBlock<T_blockSize, T_pageSize>* accessBlock,
        uint32_t const chunkSize,
        void** result,
        uint32_t const size) const -> void
    {
        std::generate(
            result,
            result + size,
            [&acc, accessBlock, chunkSize]()
            {
                void* pointer{nullptr};
                while(pointer == nullptr)
                {
                    pointer = accessBlock->create(acc, chunkSize);
                }
                return pointer;
            });
    }
};

struct ContentGenerator
{
    uint32_t counter{0U};

    ALPAKA_FN_ACC auto operator()() -> uint32_t
    {
        return counter++;
    }
};

template<typename T>
struct span
{
    T* pointer;
    size_t size;

    ALPAKA_FN_ACC auto operator[](size_t index) -> T&
    {
        return pointer[index];
    }
};

ALPAKA_FN_ACC auto forAll(auto const& acc, auto size, auto functor)
{
    auto const idx0 = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
    auto const numElements = alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0];
    for(auto i = 0; i < numElements; ++i)
    {
        auto idx = idx0 + i;
        if(idx < size)
        {
            functor(idx);
        }
    }
}

struct Create
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, auto* accessBlock, span<void*> pointers, auto chunkSize) const
    {
        forAll(acc, pointers.size, [&](auto idx) { pointers[idx] = accessBlock->create(acc, chunkSize); });
    };

    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, auto* accessBlock, span<void*> pointers, auto* chunkSizes) const
    {
        forAll(acc, pointers.size, [&](auto idx) { pointers[idx] = accessBlock->create(acc, chunkSizes[idx]); });
    };
};

struct CreateUntilSuccess
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, auto* accessBlock, span<void*> pointers, auto chunkSize) const
    {
        forAll(
            acc,
            pointers.size,
            [&](auto idx)
            {
                while(pointers[idx] == nullptr)
                {
                    pointers[idx] = accessBlock->create(acc, chunkSize);
                }
            });
    };
};


struct Destroy
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, auto* accessBlock, span<void*> pointers) const
    {
        forAll(acc, pointers.size, [&](auto idx) { accessBlock->destroy(acc, pointers[idx]); });
    };
};

struct IsValid
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        auto* accessBlock,
        void** pointers,
        bool* results,
        size_t const size) const
    {
        std::span<void*> tmpPointers(pointers, size);
        std::span<bool> tmpResults(results, size);
        std::transform(
            std::begin(tmpPointers),
            std::end(tmpPointers),
            std::begin(tmpResults),
            [&acc, accessBlock](auto pointer) { return accessBlock->isValid(acc, pointer); });
    }
};


using Host = alpaka::AccCpuSerial<Dim, Idx>;

template<typename TElem, typename TDevHost, typename TDevAcc>
struct Buffer
{
    TDevAcc m_devAcc;
    TDevHost m_devHost;

    alpaka::Vec<Dim, Idx> m_extents;

    alpaka::Buf<TDevAcc, TElem, Dim, Idx> m_onDevice;
    alpaka::Buf<TDevHost, TElem, Dim, Idx> m_onHost;

    Buffer(TDevHost const& devHost, TDevAcc const& devAcc, auto extents)
        : m_devAcc{devAcc}
        , m_devHost{devHost}
        , m_extents{extents}
        , m_onDevice(alpaka::allocBuf<TElem, Idx>(devAcc, m_extents))
        , m_onHost(alpaka::allocBuf<TElem, Idx>(devHost, m_extents))
    {
    }
};

template<typename TElem, typename TDevHost, typename TDevAcc>
auto makeBuffer(TDevHost const& devHost, TDevAcc const& devAcc, auto extents)
{
    return Buffer<TElem, TDevHost, TDevAcc>{devHost, devAcc, extents};
}

auto createChunkSizes(auto const& devHost, auto const& devAcc, auto& queue)
{
    auto chunkSizes = makeBuffer<uint32_t>(devHost, devAcc, 2U);
    chunkSizes.m_onHost[0] = 32U;
    chunkSizes.m_onHost[1] = 512U;
    alpaka::memcpy(queue, chunkSizes.m_onDevice, chunkSizes.m_onHost);
    return chunkSizes;
}

auto createPointers(auto const& devHost, auto const& devAcc, auto& queue, size_t const size)
{
    auto pointers = makeBuffer<void*>(devHost, devAcc, size);
    std::span<void*> tmp(alpaka::getPtrNative(pointers.m_onHost), pointers.m_extents[0]);
    std::fill(std::begin(tmp), std::end(tmp), reinterpret_cast<void*>(1U));
    alpaka::memcpy(queue, pointers.m_onDevice, pointers.m_onHost);
    return pointers;
}

template<typename TAcc>
auto setup()
{
    alpaka::Platform<TAcc> const platformAcc = {};
    alpaka::Platform<alpaka::AccCpuSerial<Dim, Idx>> const platformHost = {};
    alpaka::Dev<alpaka::Platform<TAcc>> const devAcc(alpaka::getDevByIdx(platformAcc, 0));
    alpaka::Dev<alpaka::Platform<Host>> const devHost(alpaka::getDevByIdx(platformHost, 0));
    alpaka::Queue<TAcc, alpaka::NonBlocking> queue{devAcc};
    return std::make_tuple(platformAcc, platformHost, devAcc, devHost, queue);
}

template<typename TAcc>
auto createWorkDiv(auto const& devAcc, auto const numElements) -> alpaka::WorkDivMembers<Dim, Idx>
{
    if constexpr(std::is_same_v<alpaka::AccToTag<TAcc>, alpaka::TagCpuSerial>)
    {
        return {{1U}, {1U}, {numElements}};
    }
    else
    {
        return alpaka::getValidWorkDiv<TAcc>(devAcc, {numElements}, {1U});
    }
}

template<typename TAcc>
auto fillWith(auto& queue, auto* accessBlock, auto const& chunkSize, auto& pointers)
{
    alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};
    alpaka::exec<TAcc>(
        queue,
        workDivSingleThread,
        FillWith{},
        accessBlock,
        chunkSize,
        alpaka::getPtrNative(pointers.m_onDevice),
        pointers.m_extents[0]);
    alpaka::wait(queue);
    alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
    alpaka::wait(queue);
}

template<typename TAcc>
auto fillAllButOne(auto& queue, auto* accessBlock, auto const& chunkSize, auto& pointers)
{
    fillWith<TAcc>(queue, accessBlock, chunkSize, pointers);
    auto* pointer1 = pointers.m_onHost[0];

    // Destroy exactly one pointer (i.e. the first). This is non-destructive on the actual values in
    // devPointers, so we don't need to wait for the copy before to finish.
    alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};
    alpaka::exec<TAcc>(
        queue,
        workDivSingleThread,
        Destroy{},
        accessBlock,
        span<void*>(alpaka::getPtrNative(pointers.m_onDevice), 1U));
    alpaka::wait(queue);
    return pointer1;
}

template<typename TAcc, size_t T_blockSize, uint32_t T_pageSize>
auto freeAllButOneOnFirstPage(auto& queue, AccessBlock<T_blockSize, T_pageSize>* accessBlock, auto& pointers)
{
    std::span<void*> tmp(alpaka::getPtrNative(pointers.m_onHost), pointers.m_extents[0]);
    std::sort(std::begin(tmp), std::end(tmp));
    // This points to the first chunk of page 0.
    auto* pointer1 = tmp[0];
    alpaka::wait(queue);
    alpaka::memcpy(queue, pointers.m_onDevice, pointers.m_onHost);
    alpaka::wait(queue);
    auto size = pointers.m_extents[0] / AccessBlock<T_blockSize, T_pageSize>::numPages() - 1;
    // Delete all other chunks on page 0.
    auto workDiv = createWorkDiv<TAcc>(pointers.m_devAcc, size);
    alpaka::exec<TAcc>(
        queue,
        workDiv,
        Destroy{},
        accessBlock,
        span<void*>(alpaka::getPtrNative(pointers.m_onDevice) + 1U, size));
    alpaka::wait(queue);
    return pointer1;
}
struct CheckContent
{
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto* content, span<void*> pointers, auto* results, auto chunkSize)
        const
    {
        auto const idx0 = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const numElements = alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0];
        for(auto i = 0; i < numElements; ++i)
        {
            auto idx = idx0 + i;
            if(idx < pointers.size)
            {
                auto* begin = reinterpret_cast<uint32_t*>(pointers[idx]);
                auto* end = begin + chunkSize / sizeof(uint32_t);
                results[idx] = std::all_of(begin, end, [idx, content](auto val) { return val == content[idx]; });
            }
        }
    }
};

template<typename TAcc>
auto checkContent(
    auto& devHost,
    auto& devAcc,
    auto& queue,
    auto& pointers,
    auto& content,
    auto& workDiv,
    auto const chunkSize)
{
    auto results = makeBuffer<bool>(devHost, devAcc, pointers.m_extents[0]);
    alpaka::exec<TAcc>(
        queue,
        workDiv,
        CheckContent{},
        alpaka::getPtrNative(content.m_onDevice),
        span<void*>(alpaka::getPtrNative(pointers.m_onDevice), pointers.m_extents[0]),
        alpaka::getPtrNative(results.m_onDevice),
        chunkSize);
    alpaka::wait(queue);
    alpaka::memcpy(queue, results.m_onHost, results.m_onDevice);
    alpaka::wait(queue);


    std::span<bool> tmpResults(alpaka::getPtrNative(results.m_onHost), results.m_extents[0]);
    auto writtenCorrectly = std::reduce(std::cbegin(tmpResults), std::cend(tmpResults), true, std::multiplies<bool>{});

    return writtenCorrectly;
}

struct GetAvailableSlots
{
    ALPAKA_FN_ACC auto operator()(auto const& /*acc*/, auto* accessBlock, auto chunkSize, auto* result) const
    {
        *result = accessBlock->getAvailableSlots(chunkSize);
    };
};

template<typename TAcc>
auto getAvailableSlots(auto* accessBlock, auto& queue, auto const& devHost, auto const& devAcc, auto chunkSize)
{
    alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};
    alpaka::wait(queue);
    auto result = makeBuffer<size_t>(devHost, devAcc, 1U);
    alpaka::wait(queue);
    alpaka::exec<TAcc>(
        queue,
        workDivSingleThread,
        GetAvailableSlots{},
        accessBlock,
        chunkSize,
        alpaka::getPtrNative(result.m_onDevice));
    alpaka::wait(queue);
    alpaka::memcpy(queue, result.m_onHost, result.m_onDevice);
    alpaka::wait(queue);
    auto tmp = result.m_onHost[0];
    alpaka::wait(queue);
    return tmp;
}

template<size_t T_blockSize, uint32_t T_pageSize>
auto pageIndex(AccessBlock<T_blockSize, T_pageSize>* accessBlock, auto* pointer)
{
    // This is a bit dirty: What we should do here is enqueue a kernel that calls accessBlock->pageIndex().
    // But we assume that the access block starts with the first page, so the pointer to the first page equals the
    // pointer to the access block. Not sure if this is reliable if the pointers are device pointers.
    return mallocMC::indexOf(pointer, accessBlock, T_pageSize);
}

struct FillAllUpAndWriteToThem
{
    ALPAKA_FN_ACC auto operator()(
        auto const& acc,
        auto* accessBlock,
        auto* content,
        span<void*> pointers,
        auto chunkSize) const
    {
        auto const idx0 = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const numElements = alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0];
        for(auto i = 0; i < numElements; ++i)
        {
            auto idx = idx0 + i;
            if(idx < pointers.size)
            {
                pointers[idx] = accessBlock->create(acc, chunkSize);
                auto* begin = reinterpret_cast<uint32_t*>(pointers[idx]);
                auto* end = begin + chunkSize / sizeof(uint32_t);
                std::fill(begin, end, content[idx]);
            }
        }
    }
};
struct CreateAndDestroMultipleTimes
{
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto* accessBlock, span<void*> pointers, auto chunkSize) const
    {
        forAll(
            acc,
            pointers.size,
            [&](auto idx)
            {
                pointers[idx] = nullptr;
                for(uint32_t j = 0; j < idx; ++j)
                {
                    // `.isValid()` is not thread-safe, so we use this direct assessment:
                    while(pointers[idx] == nullptr)
                    {
                        pointers[idx] = accessBlock->create(acc, chunkSize);
                    }
                    accessBlock->destroy(acc, pointers[idx]);
                    pointers[idx] = nullptr;
                }
                while(pointers[idx] == nullptr)
                {
                    pointers[idx] = accessBlock->create(acc, chunkSize);
                }
            });
    }
};
struct CreateAndDestroyWithDifferentSizes
{
    size_t num2{};
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto* accessBlock, span<void*> pointers, auto* chunkSizes) const
    {
        forAll(
            acc,
            pointers.size,
            [&](auto idx)
            {
                auto myChunkSize = idx % 2 == 1 and idx <= num2 ? chunkSizes[1] : chunkSizes[0];
                for(uint32_t j = 0; j < idx; ++j)
                {
                    // `.isValid()` is not thread-safe, so we use this direct assessment:
                    while(pointers[idx] == nullptr)
                    {
                        pointers[idx] = accessBlock->create(acc, myChunkSize);
                    }
                    accessBlock->destroy(acc, pointers[idx]);
                    pointers[idx] = nullptr;
                }
                while(pointers[idx] == nullptr)
                {
                    pointers[idx] = accessBlock->create(acc, myChunkSize);
                }
            });
    }
};
struct OversubscribedCreation
{
    uint32_t oversubscriptionFactor{};
    size_t availableSlots{};

    ALPAKA_FN_ACC auto operator()(auto const& acc, auto* accessBlock, span<void*> pointers, auto chunkSize) const
    {
        forAll(
            acc,
            pointers.size,
            [&](auto idx)
            {
                for(uint32_t j = 0; j < idx + 1; ++j)
                {
                    // `.isValid()` is not thread-safe, so we use this direct assessment:
                    while(pointers[idx] == nullptr)
                    {
                        pointers[idx] = accessBlock->create(acc, chunkSize);
                    }
                    accessBlock->destroy(acc, pointers[idx]);
                    pointers[idx] = nullptr;
                }

                // We only keep some of the memory. In particular, we keep one chunk less than is available,
                // such that threads looking for memory after we've finished can still find some.
                while(pointers[idx] == nullptr and idx > (oversubscriptionFactor - 1) * availableSlots + 1)
                {
                    pointers[idx] = accessBlock->create(acc, chunkSize);
                }
            });
    }
};

struct CreateAllChunkSizes
{
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto* accessBlock, span<void*> pointers, auto* chunkSizes) const
    {
        forAll(
            acc,
            pointers.size,
            [&](auto i)
            {
                pointers[i] = accessBlock->create(acc, 1U);

                std::span<uint32_t> tmpChunkSizes(chunkSizes, pageSize - BitMaskSize);
                for(auto chunkSize : tmpChunkSizes)
                {
                    accessBlock->destroy(acc, pointers[i]);
                    pointers[i] = nullptr;

                    // `.isValid()` is not thread-safe, so we use this direct assessment:
                    while(pointers[i] == nullptr)
                    {
                        pointers[i] = accessBlock->create(acc, chunkSize);
                    }
                }
            });
    }
};

TEMPLATE_LIST_TEST_CASE("Threaded AccessBlock", "", alpaka::EnabledAccTags)
{
    using Acc = alpaka::TagToAcc<TestType, Dim, Idx>;
    auto [platformAcc, platformHost, devAcc, devHost, queue] = setup<Acc>();
    auto accessBlockBuf = alpaka::allocBuf<MyAccessBlock, Idx>(devAcc, alpaka::Vec<Dim, Idx>{1U});
    alpaka::memset(queue, accessBlockBuf, 0x00);
    alpaka::wait(queue);
    auto* accessBlock = alpaka::getPtrNative(accessBlockBuf);
    auto const chunkSizes = createChunkSizes(devHost, devAcc, queue);
    auto pointers = createPointers(
        devHost,
        devAcc,
        queue,
        getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]));
    alpaka::wait(queue);

    SECTION("creates second memory somewhere else.")
    {
        size_t const size = 2U;
        auto const workDiv = createWorkDiv<Acc>(devAcc, size);
        alpaka::exec<Acc>(
            queue,
            workDiv,
            Create{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), size),
            chunkSizes.m_onHost[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        CHECK(pointers.m_onHost[0] != pointers.m_onHost[1]);
    }

    SECTION("creates memory of different chunk size in different pages.")
    {
        auto const workDiv = createWorkDiv<Acc>(devAcc, 2U);
        alpaka::exec<Acc>(
            queue,
            workDiv,
            Create{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), 2U),
            alpaka::getPtrNative(chunkSizes.m_onDevice));
        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        CHECK(pageIndex(accessBlock, pointers.m_onHost[0]) != pageIndex(accessBlock, pointers.m_onHost[1]));
    }

    SECTION("creates partly for insufficient memory with same chunk size.")
    {
        size_t const size = 2U;
        auto* lastFreeChunk = fillAllButOne<Acc>(queue, accessBlock, chunkSizes.m_onHost[0], pointers);

        // Okay, so here we start the actual test. The situation is the following:
        // There is a single chunk available.
        // We try to do two allocations.
        // So, we expect one to succeed and one to fail.
        auto const workDiv = createWorkDiv<Acc>(devAcc, size);
        alpaka::exec<Acc>(
            queue,
            workDiv,
            Create{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), size),
            chunkSizes.m_onHost[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        CHECK(
            ((pointers.m_onHost[0] == lastFreeChunk and pointers.m_onHost[1] == nullptr)
             or (pointers.m_onHost[1] == lastFreeChunk and pointers.m_onHost[0] == nullptr)));
    }

    SECTION("does not race between clean up and create.")
    {
        fillWith<Acc>(queue, accessBlock, chunkSizes.m_onHost[0], pointers);
        auto freePage = pageIndex(accessBlock, freeAllButOneOnFirstPage<Acc>(queue, accessBlock, pointers));

        // Now, pointer1 is the last valid pointer to page 0. Destroying it will clean up the page.
        alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};
        auto const workDiv = createWorkDiv<Acc>(devAcc, 1U);

        alpaka::exec<Acc>(
            queue,
            workDivSingleThread,
            Destroy{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), pointers.m_extents[0]));

        alpaka::exec<Acc>(
            queue,
            workDivSingleThread,
            CreateUntilSuccess{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), 1U),
            chunkSizes.m_onHost[0]);

        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        CHECK(pageIndex(accessBlock, pointers.m_onHost[0]) == freePage);
    }

    SECTION("destroys two pointers of different size.")
    {
        auto const workDiv = createWorkDiv<Acc>(devAcc, 2U);
        alpaka::exec<Acc>(
            queue,
            workDiv,
            Create{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), 2U),
            alpaka::getPtrNative(chunkSizes.m_onDevice));
        alpaka::wait(queue);

        alpaka::exec<Acc>(
            queue,
            workDiv,
            Destroy{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), 2U));
        alpaka::wait(queue);

        auto result = makeBuffer<bool>(devHost, devAcc, 2U);
        auto const workDivSingleThread = createWorkDiv<Acc>(devAcc, 1U);
        alpaka::exec<Acc>(
            queue,
            workDivSingleThread,
            IsValid{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            alpaka::getPtrNative(result.m_onDevice),
            result.m_extents[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, result.m_onHost, result.m_onDevice);
        alpaka::wait(queue);

        CHECK(not result.m_onHost[0]);
        CHECK(not result.m_onHost[1]);
    }

    SECTION("destroys two pointers of same size.")
    {
        auto const workDiv = createWorkDiv<Acc>(devAcc, 2U);
        alpaka::exec<Acc>(
            queue,
            workDiv,
            Create{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), 2U),
            chunkSizes.m_onHost[0]);
        alpaka::wait(queue);

        alpaka::exec<Acc>(
            queue,
            workDiv,
            Destroy{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), 2U));
        alpaka::wait(queue);

        auto result = makeBuffer<bool>(devHost, devAcc, 2U);
        result.m_onHost[0] = true;
        result.m_onHost[1] = true;
        alpaka::memcpy(queue, result.m_onDevice, result.m_onHost);
        alpaka::wait(queue);
        auto const workDivSingleThread = createWorkDiv<Acc>(devAcc, 1U);
        alpaka::exec<Acc>(
            queue,
            workDivSingleThread,
            IsValid{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            alpaka::getPtrNative(result.m_onDevice),
            result.m_extents[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, result.m_onHost, result.m_onDevice);
        alpaka::wait(queue);

        CHECK(not result.m_onHost[0]);
        CHECK(not result.m_onHost[1]);
    }

    SECTION("fills up all chunks in parallel and writes to them.")
    {
        auto content = makeBuffer<uint32_t>(
            devHost,
            devAcc,
            getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]));
        std::span<uint32_t> tmp(alpaka::getPtrNative(content.m_onHost), content.m_extents[0]);
        std::generate(std::begin(tmp), std::end(tmp), ContentGenerator{});
        alpaka::memcpy(queue, content.m_onDevice, content.m_onHost);
        alpaka::wait(queue);

        auto const workDiv = createWorkDiv<Acc>(devAcc, pointers.m_extents[0]);

        alpaka::exec<Acc>(
            queue,
            workDiv,
            FillAllUpAndWriteToThem{},
            accessBlock,
            alpaka::getPtrNative(content.m_onDevice),
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), pointers.m_extents[0]),
            chunkSizes.m_onHost[0]);

        alpaka::wait(queue);

        auto writtenCorrectly
            = checkContent<Acc>(devHost, devAcc, queue, pointers, content, workDiv, chunkSizes.m_onHost[0]);
        CHECK(writtenCorrectly);
    }

    SECTION("destroys all pointers simultaneously.")
    {
        auto const allSlots = getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]);
        auto const allSlotsOfDifferentSize
            = getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[1]);
        fillWith<Acc>(queue, accessBlock, chunkSizes.m_onHost[0], pointers);

        auto const workDiv = createWorkDiv<Acc>(devAcc, pointers.m_extents[0]);
        alpaka::exec<Acc>(
            queue,
            workDiv,
            Destroy{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), pointers.m_extents[0]));
        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        auto result = makeBuffer<bool>(devHost, devAcc, pointers.m_extents[0]);
        auto const workDivSingleThread = createWorkDiv<Acc>(devAcc, 1U);
        alpaka::exec<Acc>(
            queue,
            workDivSingleThread,
            IsValid{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            alpaka::getPtrNative(result.m_onDevice),
            result.m_extents[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, result.m_onHost, result.m_onDevice);
        alpaka::wait(queue);

        std::span<bool> tmpResults(alpaka::getPtrNative(result.m_onHost), result.m_extents[0]);
        CHECK(std::none_of(std::cbegin(tmpResults), std::cend(tmpResults), [](auto const val) { return val; }));

        CHECK(getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]) == allSlots);
        CHECK(
            getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[1])
            == allSlotsOfDifferentSize);
    }

    SECTION("creates and destroys multiple times.")
    {
        auto const workDiv = createWorkDiv<Acc>(devAcc, pointers.m_extents[0]);

        alpaka::exec<Acc>(
            queue,
            workDiv,
            CreateAndDestroMultipleTimes{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), pointers.m_extents[0]),
            chunkSizes.m_onHost[0]);
        alpaka::wait(queue);
        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        std::span<void*> tmpPointers(alpaka::getPtrNative(pointers.m_onHost), pointers.m_extents[0]);
        std::sort(std::begin(tmpPointers), std::end(tmpPointers));
        CHECK(std::unique(std::begin(tmpPointers), std::end(tmpPointers)) == std::end(tmpPointers));
    }

    SECTION("creates and destroys multiple times with different sizes.")
    {
        SKIP("This test appears to be brittle due to reasons described in the comments.");
        // CAUTION: This test can fail because we are currently using exactly as much space as is available but with
        // multiple different chunk sizes. That means that if one of them occupies more pages than it minimally needs,
        // the other one will lack pages to with their respective chunk size. This seems not to be a problem currently
        // but it might be more of a problem once we move to device and once we include proper scattering.

        // TODO(lenz): This could be solved by "fixing the chunk sizes beforehand", e.g., one could allocate
        // one chunk
        // on each page such that half the pages have one chunk size and the other half has the other chunk size.

        // Make sure that num2 > num1.
        auto num1 = getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]);
        auto num2 = getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[1]);
        auto totalSlots = num1 / 2 + num2 / 2;

        alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{totalSlots}, Idx{1}};

        alpaka::exec<Acc>(
            queue,
            workDiv,
            CreateAndDestroyWithDifferentSizes{num2},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), pointers.m_extents[0]),
            alpaka::getPtrNative(chunkSizes.m_onDevice));
        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        std::span<void*> tmpPointers(alpaka::getPtrNative(pointers.m_onHost), MyAccessBlock::numPages());
        std::sort(std::begin(tmpPointers), std::end(tmpPointers));
        CHECK(std::unique(std::begin(tmpPointers), std::end(tmpPointers)) == std::end(tmpPointers));
    }

    SECTION("can handle oversubscription.")
    {
        SKIP("Somehow this test is flaky. Haven't found out why yet.");
        uint32_t oversubscriptionFactor = 2U;
        auto availableSlots = getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]);

        // This is oversubscribed but we will only hold keep less than 1/oversubscriptionFactor of the memory in the
        // end.
        auto manyPointers = makeBuffer<void*>(devHost, devAcc, oversubscriptionFactor * availableSlots);
        auto workDiv = createWorkDiv<Acc>(devAcc, manyPointers.m_extents[0]);

        alpaka::wait(queue);
        alpaka::exec<Acc>(
            queue,
            workDiv,
            OversubscribedCreation{oversubscriptionFactor, availableSlots},
            accessBlock,
            span<void*>(alpaka::getPtrNative(manyPointers.m_onDevice), manyPointers.m_extents[0]),
            chunkSizes.m_onHost[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, manyPointers.m_onHost, manyPointers.m_onDevice);
        alpaka::wait(queue);

        // We only let the last (availableSlots-1) keep their memory. So, the rest at the beginning should have a
        // nullptr.
        std::span<void*> tmpManyPointers(alpaka::getPtrNative(manyPointers.m_onHost), manyPointers.m_extents[0]);
        auto beginNonNull = std::begin(tmpManyPointers) + (oversubscriptionFactor - 1) * availableSlots + 1;

        CHECK(std::all_of(
            std::begin(tmpManyPointers),
            beginNonNull,
            [](auto const pointer) { return pointer == nullptr; }));

        std::sort(beginNonNull, std::end(tmpManyPointers));
        CHECK(std::unique(beginNonNull, std::end(tmpManyPointers)) == std::end(tmpManyPointers));
    }

    SECTION("can handle many different chunk sizes.")
    {
        auto chunkSizes = makeBuffer<uint32_t>(devHost, devAcc, pageSize - BitMaskSize);
        std::span<uint32_t> tmp(alpaka::getPtrNative(chunkSizes.m_onHost), chunkSizes.m_extents[0]);
        std::iota(std::begin(tmp), std::end(tmp), 1U);
        alpaka::memcpy(queue, chunkSizes.m_onDevice, chunkSizes.m_onHost);
        alpaka::wait(queue);

        auto workDiv = createWorkDiv<Acc>(devAcc, MyAccessBlock::numPages());

        alpaka::exec<Acc>(
            queue,
            workDiv,
            CreateAllChunkSizes{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), MyAccessBlock::numPages()),
            alpaka::getPtrNative(chunkSizes.m_onDevice));

        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        std::span<void*> tmpPointers(alpaka::getPtrNative(pointers.m_onHost), MyAccessBlock::numPages());
        std::sort(std::begin(tmpPointers), std::end(tmpPointers));
        CHECK(std::unique(std::begin(tmpPointers), std::end(tmpPointers)) == std::end(tmpPointers));
    }
    alpaka::wait(queue);
}

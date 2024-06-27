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
#include "mallocMC/creationPolicies/Scatter/Hash.hpp"
#include "mallocMC/creationPolicies/Scatter/PageInterpretation.hpp"
#include "mallocMC/mallocMC_utils.hpp"

#include <algorithm>
#include <alpaka/atomic/AtomicAtomicRef.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/idx/Accessors.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/mem/fence/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <vector>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    constexpr const uint32_t pageTableEntrySize = 4U + 4U;

    template<size_t T_numPages>
    struct PageTable
    {
        uint32_t _chunkSizes[T_numPages]{};
        uint32_t _fillingLevels[T_numPages]{};
    };

    template<size_t T_blockSize, uint32_t T_pageSize>
    class AccessBlock
    {
    public:
        ALPAKA_FN_ACC [[nodiscard]] constexpr static auto numPages() -> size_t
        {
            return T_blockSize / (T_pageSize + pageTableEntrySize);
        }

        ALPAKA_FN_ACC [[nodiscard]] auto getAvailableSlots(uint32_t const chunkSize) -> size_t
        {
            if(chunkSize < T_pageSize)
            {
                return getAvailableChunks(chunkSize);
            }
            return getAvailableMultiPages(chunkSize);
        }

        ALPAKA_FN_ACC auto pageIndex(void* pointer) const -> ssize_t
        {
            return indexOf(pointer, pages, T_pageSize);
        }

        template<typename TAcc>
        ALPAKA_FN_ACC auto isValid(TAcc const& acc, void* pointer) -> bool
        {
            if(pointer == nullptr)
            {
                return false;
            }
            auto const index = pageIndex(pointer);
            auto chunkSize = atomicLoad(acc, pageTable._chunkSizes[index]);
            if(chunkSize >= T_pageSize)
            {
                return true;
            }
            return chunkSize == 0U or atomicLoad(acc, pageTable._fillingLevels[index]) == 0U
                ? false
                : interpret(index, chunkSize).isValid(acc, pointer);
        }

        template<typename TAcc>
        ALPAKA_FN_ACC auto create(TAcc const& acc, uint32_t const numBytes) -> void*
        {
            void* pointer{nullptr};
            if(numBytes >= multiPageThreshold())
            {
                pointer = createOverMultiplePages(numBytes);
            }
            else
            {
                pointer = createChunk(acc, numBytes);
            }
            return pointer;
        }

        template<typename TAcc>
        ALPAKA_FN_ACC auto destroy(TAcc const& acc, void* const pointer) -> void
        {
            auto const index = pageIndex(pointer);
            if(index > static_cast<ssize_t>(numPages()) || index < 0)
            {
#ifndef NDEBUG
                throw std::runtime_error{
                    "Attempted to destroy an invalid pointer! Pointer does not point to any page."};
#endif // NDEBUG
                return;
            }
            auto const chunkSize = atomicLoad(acc, pageTable._chunkSizes[index]);
            if(chunkSize >= multiPageThreshold())
            {
                destroyOverMultiplePages(index, chunkSize);
            }
            else
            {
                destroyChunk(acc, pointer, index, chunkSize);
            }
        }

    private:
        DataPage<T_pageSize> pages[numPages()]{};
        PageTable<numPages()> pageTable{};

        ALPAKA_FN_ACC constexpr static auto multiPageThreshold() -> uint32_t
        {
            return T_pageSize - sizeof(BitMaskStorageType<>);
        }

        ALPAKA_FN_ACC auto interpret(size_t const pageIndex, uint32_t const chunkSize)
        {
            return PageInterpretation<T_pageSize>(pages[pageIndex], chunkSize);
        }

        ALPAKA_FN_ACC [[nodiscard]] auto getAvailableChunks(uint32_t const chunkSize) const -> size_t
        {
            // This is not thread-safe!
            return std::transform_reduce(
                std::cbegin(pageTable._chunkSizes),
                std::cend(pageTable._chunkSizes),
                std::cbegin(pageTable._fillingLevels),
                0U,
                std::plus<size_t>{},
                [chunkSize](auto const localChunkSize, auto const fillingLevel)
                {
                    return chunkSize == localChunkSize or localChunkSize == 0U
                        ? PageInterpretation<T_pageSize>::numChunks(chunkSize) - fillingLevel
                        : 0U;
                });
        }

        ALPAKA_FN_ACC [[nodiscard]] auto getAvailableMultiPages(uint32_t const chunkSize) -> size_t
        {
            // This is the most inefficient but simplest and only thread-safe way I could come up with. If we ever
            // need this in production, we might want to revisit this.
            void* pointer = nullptr;
            std::vector<void*> pointers;
            while((pointer = createOverMultiplePages(chunkSize)))
            {
                pointers.push_back(pointer);
            }
            // TODO(lenz): This is a dirty hack! There is a template TAcc and I provide something explicit. This only
            // works because multiple pages are not yet thread-safe.
            std::for_each(
                std::begin(pointers),
                std::end(pointers),
                [&](auto ptr) { destroyOverMultiplePages(pageIndex(ptr), chunkSize); });
            return pointers.size();
        }

        ALPAKA_FN_ACC auto createOverMultiplePages(uint32_t const numBytes) -> void*
        {
            auto numPagesNeeded = ceilingDivision(numBytes, T_pageSize);
            if(numPagesNeeded > numPages())
            {
                return nullptr;
            }
            auto chunk = createContiguousPages(numPagesNeeded);
            if(chunk)
            {
                for(uint32_t i = 0; i < numPagesNeeded; ++i)
                {
                    pageTable._chunkSizes[chunk.value().index + i] = numBytes;
                    pageTable._fillingLevels[chunk.value().index + i] = 1U;
                }
                return chunk.value().pointer;
            }
            return nullptr;
        }

        ALPAKA_FN_ACC static auto noFreePageFound()
        {
            return numPages();
        }

        ALPAKA_FN_ACC static auto startIndex(auto const& acc)
        {
            return (laneid() * alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc).sum()) % numPages();
        }

        template<typename TAcc>
        ALPAKA_FN_ACC auto createChunk(TAcc const& acc, uint32_t const numBytes) -> void*
        {
            auto index = startIndex(acc);

            // Under high pressure, this loop could potentially run for a long time because the information where and
            // when we started our search is not maintained and/or used. This is a feature, not a bug: Given a
            // consistent state, the loop will terminate once a free chunk is found or when all chunks are filled for
            // long enough that `choosePage` could verify that each page is filled in a single run.
            //
            // The seemingly non-terminating behaviour that we wrap around multiple times can only occur (assuming a
            // consistent, valid state of the data) when there is high demand for memory such that pages that appear
            // free to `choosePage` are repeatedly found but then the free chunks are scooped away by other threads.
            //
            // In the latter case, it is considered desirable to wrap around multiple times until the thread was fast
            // enough to acquire some memory.
            index = choosePage(acc, numBytes, index);
            void* pointer = index != noFreePageFound()
                ? PageInterpretation<T_pageSize>{pages[index], numBytes}.create(acc)
                : nullptr;

            while(index != noFreePageFound() and pointer == nullptr)
            {
                leavePage(acc, index);
                ++index;
                index = choosePage(acc, numBytes, index);
                pointer = PageInterpretation<T_pageSize>{pages[index], numBytes}.create(acc);
            }

            return pointer;
        }

        template<typename TAcc>
        ALPAKA_FN_ACC auto choosePage(TAcc const& acc, uint32_t const numBytes, size_t const startIndex = 0) -> size_t
        {
            return wrappingLoop(
                acc,
                startIndex,
                numPages(),
                noFreePageFound(),
                [this, numBytes](auto const& localAcc, auto const index)
                { return thisPageIsAppropriate(localAcc, index, numBytes) ? index : noFreePageFound(); });
        }

        template<typename TAcc>
        ALPAKA_FN_ACC auto thisPageIsAppropriate(TAcc const& acc, size_t const index, uint32_t const numBytes) -> bool
        {
            bool appropriate = false;
            if(enterPage(acc, index, numBytes))
            {
                auto oldChunkSize = atomicCAS(acc, pageTable._chunkSizes[index], 0U, numBytes);
                appropriate = (oldChunkSize == 0U || oldChunkSize == numBytes);
            }
            if(not appropriate)
            {
                leavePage(acc, index);
            }
            return appropriate;
        }

        ALPAKA_FN_ACC auto createContiguousPages(uint32_t const numPagesNeeded) -> std::optional<Chunk>
        {
            for(size_t index = 0; index < numPages() - (numPagesNeeded - 1); ++index)
            {
                if(std::all_of(
                       &pageTable._chunkSizes[index],
                       &pageTable._chunkSizes[index + numPagesNeeded],
                       [](auto const val) { return val == 0U; }))
                {
                    return std::optional<Chunk>({static_cast<uint32_t>(index), &pages[index]});
                };
            }
            return std::nullopt;
        }

        template<typename TAcc>
        ALPAKA_FN_ACC void destroyChunk(
            TAcc const& acc,
            void* pointer,
            uint32_t const pageIndex,
            uint32_t const chunkSize)
        {
            auto page = interpret(pageIndex, chunkSize);
            page.destroy(acc, pointer);
            leavePage(acc, pageIndex);
        }

        template<typename TAcc>
        ALPAKA_FN_ACC auto enterPage(TAcc const& acc, uint32_t const pageIndex, uint32_t const expectedChunkSize)
            -> bool
        {
            auto const oldFilling = atomicAdd(acc, pageTable._fillingLevels[pageIndex], 1U);
            // We assume that this page has the correct chunk size. If not, the chunk size is either 0 (and oldFilling
            // must be 0, too) or the next check will fail.
            return oldFilling < PageInterpretation<T_pageSize>::numChunks(expectedChunkSize);
        }

        template<typename TAcc>
        ALPAKA_FN_ACC void leavePage(TAcc const& acc, uint32_t const pageIndex)
        {
            // This outermost atomicSub is an optimisation: We can fast-track this if we are not responsible for the
            // clean-up. Using 0U -> 1U in the atomicCAS and comparison further down would have the same effect (if the
            // else branch contained the simple subtraction). It's a matter of which case shall have one operation
            // less.
            if(atomicSub(acc, pageTable._fillingLevels[pageIndex], 1U) == 1U)
            {
                // This number depends on the chunk size which will at some point get reset to 0 and might even get set
                // to another value by another thread before our task is complete here. Naively, one could expect this
                // "weakens" the lock and makes it insecure. But not the case and having it like this is a feature, not
                // a bug, as is proven in the comments below.
                auto lock = T_pageSize;
                auto latestFilling = atomicCAS(acc, pageTable._fillingLevels[pageIndex], 0U, lock);
                if(latestFilling == 0U)
                {
                    auto chunkSize = atomicLoad(acc, pageTable._chunkSizes[pageIndex]);
                    if(chunkSize != 0)
                    {
                        // At this point it's guaranteed that the fiilling level is numChunks and thereby locked.
                        // Furthermore, chunkSize cannot have changed because we maintain the invariant that the
                        // filling level is always considered first, so no other thread can have passed that barrier to
                        // reset it.
                        PageInterpretation<T_pageSize>{pages[pageIndex], chunkSize}.cleanup();
                        alpaka::mem_fence(acc, alpaka::memory_scope::Device{});

                        // It is important to keep this after the clean-up line above: Otherwise another thread with a
                        // smaller chunk size might circumvent our lock and already start allocating before we're done
                        // cleaning up.
                        atomicCAS(acc, pageTable._chunkSizes[pageIndex], chunkSize, 0U);
                    }

                    // TODO(lenz): Original version had a thread fence at this point in order to invalidate potentially
                    // cached bit masks. Check if that's necessary!

                    // At this point, there might already be another thread (with another chunkSize) on this page but
                    // that's fine. It won't see the full capacity but we can just subtract what we've added before:
                    atomicSub(acc, pageTable._fillingLevels[pageIndex], lock);
                }
            }
        }

        ALPAKA_FN_ACC void destroyOverMultiplePages(size_t const pageIndex, uint32_t const chunkSize)
        {
            auto numPagesNeeded = ceilingDivision(chunkSize, T_pageSize);
            for(uint32_t i = 0; i < numPagesNeeded; ++i)
            {
                pageTable._chunkSizes[pageIndex + i] = 0U;
                pageTable._fillingLevels[pageIndex + i] = 0U;
            }
        }
    };

    template<size_t T_blockSize, uint32_t T_pageSize>
    struct Heap
    {
        size_t heapSize{};
        AccessBlock<T_blockSize, T_pageSize>* accessBlocks{};

        ALPAKA_FN_ACC [[nodiscard]] auto numBlocks() const -> size_t
        {
            return heapSize / T_blockSize;
        }

        ALPAKA_FN_ACC [[nodiscard]] auto noFreeBlockFound() const -> size_t
        {
            return numBlocks();
        }

        ALPAKA_FN_ACC auto startIndex(auto const& acc, uint32_t const numBytes) const
        {
            return (numBytes * alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc).sum()) % numBlocks();
        }

        template<typename AlignmentPolicy, typename AlpakaAcc>
        ALPAKA_FN_ACC auto create(const AlpakaAcc& acc, uint32_t bytes) -> void*
        {
            return wrappingLoop(
                acc,
                startIndex(acc, bytes),
                numBlocks(),
                static_cast<void*>(nullptr),
                [this, bytes](auto const& localAcc, auto const index)
                { return accessBlocks[index].create(localAcc, bytes); });
        }

        template<typename AlpakaAcc>
        ALPAKA_FN_ACC auto destroy(const AlpakaAcc& acc, void* pointer) -> void
        {
            auto blockIndex = indexOf(pointer, accessBlocks, T_blockSize);
            accessBlocks[blockIndex].destroy(acc, pointer);
        }
    };

} // namespace mallocMC::CreationPolicies::ScatterAlloc

struct InitKernel
{
    template<size_t T_blockSize, uint32_t T_pageSize>
    ALPAKA_FN_ACC auto operator()(
        auto const& /*unused*/,
        mallocMC::CreationPolicies::ScatterAlloc::Heap<T_blockSize, T_pageSize>* m_heap,
        void* m_heapmem,
        size_t const m_memsize) const
    {
        m_heap->accessBlocks
            = static_cast<mallocMC::CreationPolicies::ScatterAlloc::AccessBlock<T_blockSize, T_pageSize>*>(m_heapmem);
        m_heap->heapSize = m_memsize;
    }
};

namespace mallocMC::CreationPolicies
{

    template<typename T_HeapConfig, typename T_HashConfig>
    struct Scatter : public ScatterAlloc::Heap<T_HeapConfig::accessblocksize, T_HeapConfig::pagesize>
    {
        static_assert(T_HeapConfig::resetfreedpages, "resetfreedpages = false is no longer implemented.");
        static_assert(T_HeapConfig::wastefactor == 1, "A wastefactor is no yet implemented.");

        ALPAKA_FN_ACC static auto isOOM(void* pointer, uint32_t const /*unused size*/) -> bool
        {
            return pointer == nullptr;
        }

        template<typename TAcc>
        static void initHeap(auto& dev, auto& queue, auto* heap, void* pool, size_t memsize)
        {
            using Dim = typename alpaka::trait::DimType<TAcc>::type;
            using Idx = typename alpaka::trait::IdxType<TAcc>::type;
            using VecType = alpaka::Vec<Dim, Idx>;

            auto poolView = alpaka::createView(dev, reinterpret_cast<char*>(pool), alpaka::Vec<Dim, Idx>(memsize));
            alpaka::memset(queue, poolView, 0U);
            alpaka::wait(queue);

            auto workDivSingleThread
                = alpaka::WorkDivMembers<Dim, Idx>{VecType::ones(), VecType::ones(), VecType::ones()};
            alpaka::exec<TAcc>(queue, workDivSingleThread, InitKernel{}, heap, pool, memsize);
            alpaka::wait(queue);
        }

        constexpr const static bool providesAvailableSlots = false;
        static auto classname() -> std::string
        {
            return "Scatter";
        }
    };
} // namespace mallocMC::CreationPolicies

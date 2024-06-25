/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de

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

#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <cassert>
#include <iostream>
#include <mallocMC/mallocMC.hpp>
#include <numeric>

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

// Define the device accelerator
using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

struct ScatterHeapConfig
{
    static constexpr auto heapsize = 2U * 1024U * 1024U * 1024U;
    static constexpr auto accessblocksize = 2U * 1024U * 1024U * 1024U;
    static constexpr auto pagesize = 4096;
    static constexpr auto regionsize = 16;
    static constexpr auto wastefactor = 1;
    static constexpr auto resetfreedpages = true;
};

struct ScatterHashConfig
{
    static constexpr auto hashingK = 38183;
    static constexpr auto hashingDistMP = 17497;
    static constexpr auto hashingDistWP = 1;
    static constexpr auto hashingDistWPRel = 1;
};

struct XMallocConfig
{
    static constexpr auto pagesize = ScatterHeapConfig::pagesize;
};

struct ShrinkConfig
{
    static constexpr auto dataAlignment = 16;
};

using ScatterAllocator = mallocMC::Allocator<
    Acc,
    mallocMC::CreationPolicies::Scatter<ScatterHeapConfig, ScatterHashConfig>,
    mallocMC::DistributionPolicies::Noop,
    mallocMC::OOMPolicies::ReturnNull,
    mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
    mallocMC::AlignmentPolicies::Shrink<ShrinkConfig>>;

ALPAKA_STATIC_ACC_MEM_GLOBAL int** arA;
ALPAKA_STATIC_ACC_MEM_GLOBAL int** arB;
ALPAKA_STATIC_ACC_MEM_GLOBAL int** arC;

auto main() -> int
{
    constexpr auto length = 100;

    auto const platform = alpaka::Platform<Acc>{};
    const auto dev = alpaka::getDevByIdx(platform, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};

    auto const devProps = alpaka::getAccDevProps<Acc>(dev);
    unsigned const block = std::min(static_cast<size_t>(32U), static_cast<size_t>(devProps.m_blockThreadCountMax));

    // round up
    auto grid = (length + block - 1U) / block;
    assert(length <= block * grid); // necessary for used algorithm

    // init the heap
    std::cerr << "initHeap...";
    auto const heapSize = 2U * 1024U * 1024U * 1024U;
    ScatterAllocator scatterAlloc(dev, queue, heapSize); // 1GB for device-side malloc
    std::cerr << "done\n";
    std::cout << ScatterAllocator::info("\n") << '\n';

    // create arrays of arrays on the device
    {
        auto createArrayPointers
            = [] ALPAKA_FN_ACC(const Acc& acc, int x, int y, ScatterAllocator::AllocatorHandle allocHandle)
        {
            arA<Acc> = static_cast<int**>(allocHandle.malloc(acc, sizeof(int*) * x * y));
            arB<Acc> = static_cast<int**>(allocHandle.malloc(acc, sizeof(int*) * x * y));
            arC<Acc> = static_cast<int**>(allocHandle.malloc(acc, sizeof(int*) * x * y));
        };
        const auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}};
        alpaka::enqueue(
            queue,
            alpaka::createTaskKernel<Acc>(
                workDiv,
                createArrayPointers,
                grid,
                block,
                scatterAlloc.getAllocatorHandle()));
    }

    // fill 2 of them all with ascending values
    {
        auto fillArrays
            = [] ALPAKA_FN_ACC(const Acc& acc, int localLength, ScatterAllocator::AllocatorHandle allocHandle)
        {
            const auto id = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

            arA<Acc>[id] = static_cast<int*>(allocHandle.malloc(acc, localLength * sizeof(int)));
            arB<Acc>[id] = static_cast<int*>(allocHandle.malloc(acc, localLength * sizeof(int)));
            arC<Acc>[id] = static_cast<int*>(allocHandle.malloc(acc, localLength * sizeof(int)));

            for(int i = 0; i < localLength; ++i)
            {
                arA<Acc>[id][i] = static_cast<int>(id * localLength + i);
                arB<Acc>[id][i] = static_cast<int>(id * localLength + i);
            }
        };
        const auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{grid}, Idx{block}, Idx{1}};
        alpaka::enqueue(
            queue,
            alpaka::createTaskKernel<Acc>(workDiv, fillArrays, length, scatterAlloc.getAllocatorHandle()));
    }

    // add the 2 arrays (vector addition within each thread)
    // and do a thread-wise reduce to sums
    {
        auto sumsBufferAcc = alpaka::allocBuf<int, Idx>(dev, Idx{block * grid});

        auto addArrays = [] ALPAKA_FN_ACC(const Acc& acc, int localLength, int* sums)
        {
            const auto id = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

            sums[id] = 0;
            for(int i = 0; i < localLength; ++i)
            {
                arC<Acc>[id][i] = arA<Acc>[id][i] + arB<Acc>[id][i];
                sums[id] += arC<Acc>[id][i];
            }
        };
        const auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{grid}, Idx{block}, Idx{1}};
        alpaka::enqueue(
            queue,
            alpaka::createTaskKernel<Acc>(workDiv, addArrays, length, alpaka::getPtrNative(sumsBufferAcc)));

        auto const platformCPU = alpaka::Platform<alpaka::DevCpu>{};
        const auto hostDev = alpaka::getDevByIdx(platformCPU, 0);

        auto sumsBufferHost = alpaka::allocBuf<int, Idx>(hostDev, Idx{block * grid});
        alpaka::memcpy(queue, sumsBufferHost, sumsBufferAcc, Idx{block * grid});
        alpaka::wait(queue);

        const auto* sumsPtr = alpaka::getPtrNative(sumsBufferHost);
        const auto sum = std::accumulate(sumsPtr, sumsPtr + block * grid, size_t{0});
        std::cout << "The sum of the arrays on GPU is " << sum << '\n';
    }

    const auto n = static_cast<size_t>(block * grid * length);
    const auto gaussian = n * (n - 1);
    std::cout << "The gaussian sum as comparison: " << gaussian << '\n';

    /*constexpr*/ if(mallocMC::Traits<ScatterAllocator>::providesAvailableSlots)
    {
        std::cout << "there are ";
        std::cout << scatterAlloc.getAvailableSlots(dev, queue, 1024U * 1024U);
        std::cout << " Slots of size 1MB available\n";
    }

    {
        auto freeArrays = [] ALPAKA_FN_ACC(const Acc& acc, ScatterAllocator::AllocatorHandle allocHandle)
        {
            const auto id = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
            allocHandle.free(acc, arA<Acc>[id]);
            allocHandle.free(acc, arB<Acc>[id]);
            allocHandle.free(acc, arC<Acc>[id]);
        };
        const auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{grid}, Idx{block}, Idx{1}};
        alpaka::enqueue(queue, alpaka::createTaskKernel<Acc>(workDiv, freeArrays, scatterAlloc.getAllocatorHandle()));
    }

    {
        auto freeArrayPointers = [] ALPAKA_FN_ACC(const Acc& acc, ScatterAllocator::AllocatorHandle allocHandle)
        {
            allocHandle.free(acc, arA<Acc>);
            allocHandle.free(acc, arB<Acc>);
            allocHandle.free(acc, arC<Acc>);
        };
        const auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}};
        alpaka::enqueue(
            queue,
            alpaka::createTaskKernel<Acc>(workDiv, freeArrayPointers, scatterAlloc.getAllocatorHandle()));
    }

    return 0;
}

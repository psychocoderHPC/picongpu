/**
 * Copyright 2015 Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once


#include "types.h"
#include "nvidia/warp.hpp"
#include <math_functions.h>
#include <device_functions.h>
#include <cub/cub.cuh>
#include <cub/util_ptx.cuh>


namespace PMacc
{
namespace nvidia
{

/** optimized atomic increment
 *
 * - only optimized if PTX ISA >=3.0
 * - this atomic uses warp aggregation to speedup the operation compared to
 *   cuda `atomicInc()`
 * - cuda `atomicAdd()` is used if the compute architecture not supports
 *   warp aggregation
 *   is used
 * - all participate threads must change the same
 *   pointer (ptr) else the result is unspecified
 *
 * @param ptr pointer to memory (must be the same address for all threads in a block)
 *
 * This warp aggregated atomic increment implementation based on
 * nvidia parallel forall example
 * http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
 */
DINLINE
int atomicAllInc(int *ptr)
{
#if (__CUDA_ARCH__ >= 300)
    const int mask = __ballot(1);
    /* select the leader */
    const int leader = __ffs(mask) - 1;
    int restult;
    const int lanId = getLaneId();
    if (lanId == leader)
        restult = atomicAdd(ptr, __popc(mask));
    restult = warpBroadcast(restult, leader);
    /* each thread computes its own value */
    return restult + __popc(mask & ((1 << lanId) - 1));
#else
    return atomicAdd(ptr, 1);
#endif
}

/** optimized atomic value exchange
 *
 * - only optimized if PTX ISA >=2.0
 * - this atomic uses warp vote function to speedup the operation
 *   compared to cuda `atomicExch()`
 * - cuda `atomicExch()` is used if the compute architecture not supports
 *   warps vote functions
 * - all participate threads must change the same
 *   pointer (ptr) and set the same value, else the
 *   result is unspecified
 *
 * @param ptr pointer to memory (must be the same address for all threads in a block)
 * @param value new value (must be the same for all threads in a block)
 */
template<typename T_Type>
DINLINE void
atomicAllExch(T_Type* ptr, const T_Type value)
{
#if (__CUDA_ARCH__ >= 200)
    const int mask = __ballot(1);
    // select the leader
    const int leader = __ffs(mask) - 1;
    // leader does the update
    if (getLaneId() == leader)
#endif
        ::atomicExch(ptr, value);
}

#if (__CUDA_ARCH__ >= 300)

/**
 * This in warp peer search based on
 * "Voting and Shuffling to Optimize Atomic Operations" (nvidia parallel forall)
 * http://devblogs.nvidia.com/parallelforall/voting-and-shuffling-optimize-atomic-operations/
 * Editor:  Elmar Westphal (version from Aug 6th 2015)
 */
template<typename G>
DINLINE uint32_t get_peers(G my_key)
{
    uint32_t peers = 0;
    bool is_peer;
    uint32_t unclaimed = __ballot(1); // in the beginning, no threads are claimed
    do
    {
        G other_key = cub::ShuffleIndex(my_key, __ffs(unclaimed) - 1); // get key from least unclaimed lane
        is_peer = (my_key == other_key); // do we have a match?
        peers = __ballot(is_peer); // find all matches
        unclaimed ^= peers; // matches are no longer unclaimed
    }
    while (!is_peer); // repeat as long as we haven’t found our match
    return peers;
}
#endif

namespace detail
{

template<typename T_Type>
DINLINE void atomicAdd(T_Type* inAddress, T_Type value)
{
    ::atomicAdd(inAddress, value);
}

DINLINE void atomicAdd(double* inAddress, double value)
{
    uint64_cu* address = (uint64_cu*) inAddress;
    double old = value;
    while (
           (old = __longlong_as_double(::atomicExch(address,
                                                  (uint64_cu) __double_as_longlong(__longlong_as_double(atomicExch(address, (uint64_cu) 0L)) +
                                                                                   old)))) != 0.0);
}

} //namespace detail

#if (__CUDA_ARCH__ >= 300)

/**
 * This in warp peer add based on
 * "Voting and Shuffling to Optimize Atomic Operations" (nvidia parallel forall)
 * http://devblogs.nvidia.com/parallelforall/voting-and-shuffling-optimize-atomic-operations/
 * Editor:  Elmar Westphal (version from Aug 6th 2015)
 */
template <typename F>
DINLINE void add_peers(F *dest, F x, uint32_t peers)
{
    int lane = getLaneId();
    int first = __ffs(peers) - 1; // find the leader
    uint32_t rel_pos = __popc(peers << (32 - lane)); // find our own place
    peers &= (0xfffffffe << lane); // drop everything to our right
    while (__any(peers))
    { // stay alive as long as anyone is working
        int next = __ffs(peers); // find out what to add
        F t = cub::ShuffleIndex(x, next - 1); // get what to add (undefined if nothing)
        if (next) // important: only add if there really is anything
            x += t;
        uint32_t done = rel_pos & 1; // local data was used in iteration when its LSB is set
        peers &= __ballot(!done); // clear out all peers that were just used
        rel_pos >>= 1; // count iterations by shifting position
    }
    if (lane == first) // only leader threads for each key perform atomics
        detail::atomicAdd(dest, x);
    // F res = cub::ShuffleIndex(x, first); // distribute result (if needed)
    //return res; // may also return x or return value of atomic, as needed
}
#endif

} //namespace nvidia
} //namespace PMacc

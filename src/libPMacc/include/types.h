/**
 * Copyright 2013-2015 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz
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

#include "debug/PMaccVerbose.hpp"

#include <alpaka/alpaka.hpp>

#define BOOST_MPL_LIMIT_VECTOR_SIZE 20
#define BOOST_MPL_LIMIT_MAP_SIZE 20
#include <boost/typeof/std/utility.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/filesystem.hpp>

#include <stdint.h>
#include <stdexcept>

#define PMACC_AUTO_TPL(var,...) BOOST_AUTO_TPL(var,(__VA_ARGS__))
#define PMACC_AUTO(var,...) BOOST_AUTO(var,(__VA_ARGS__))

namespace PMacc
{
    using AlpakaSize = size_t;
    using AlpakaIdxSize = int;

#if !(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__))
    #define PMACC_ACC_CPU
#endif

    using AlpakaHostDev = alpaka::dev::DevCpu;
#ifdef PMACC_ACC_CPU
    using AlpakaAccDev = alpaka::dev::DevCpu;
    using AlpakaAccStream = alpaka::stream::StreamCpuAsync;
    //using AlpakaAccStream = alpaka::stream::StreamCpuSync;
    template<
        typename TDim>
    using AlpakaAcc = alpaka::acc::AccCpuOmp2Threads<TDim, AlpakaIdxSize>;
    //using AlpakaAcc = alpaka::acc::AccCpuOmp2Blocks<TDim, AlpakaIdxSize>;
#else
    using AlpakaAccDev = alpaka::dev::DevCudaRt;
    using AlpakaAccStream = alpaka::stream::StreamCudaRtAsync;
    template<
        typename TDim
    >
    using AlpakaAcc = alpaka::acc::AccGpuCudaRt<TDim, AlpakaIdxSize>;
#endif

namespace bmpl = boost::mpl;
namespace bfs = boost::filesystem;

//short name for access verbose types of libPMacc
typedef PMaccVerbose ggLog;

typedef uint64_t id_t;
typedef unsigned long long int uint64_cu;
typedef long long int int64_cu;

#define PMACC_NO_NVCC_HDWARNING ALPAKA_NO_HOST_ACC_WARNING
#define HDINLINE ALPAKA_FN_HOST_ACC
#define DINLINE ALPAKA_FN_ACC
#define HINLINE ALPAKA_FN_HOST

/**
 * Bitmask which describes the direction of communication.
 *
 * Bitmasks may be combined logically, e.g. LEFT+TOP = TOPLEFT.
 * It is not possible to combine complementary masks (e.g. FRONT and BACK),
 * as a bitmask always defines one direction of communication (send or receive).
 */
enum ExchangeType
{
    RIGHT = 1u, LEFT = 2u, BOTTOM = 3u, TOP = 6u, BACK = 9u, FRONT = 18u // 3er-System
};

/**
 * Defines number of dimensions (1-3)
 */

#define DIM1 1u
#define DIM2 2u
#define DIM3 3u

/**
 * Internal event/task type used for notifications in the event system.
 */
enum EventType
{
    FINISHED, COPYHOST2DEVICE, COPYDEVICE2HOST, COPYDEVICE2DEVICE, SENDFINISHED, RECVFINISHED, LOGICALAND, SETVALUE, GETVALUE, KERNEL
};

/**
 * Alignment macros.
 *
 * You must align all array and structs which can used on device!
 *
 * We use __VA_ARGS__ here even though we only ever allow one type!
 * This allows types as argument that contain commas which would not be possible else.
 */
#define PMACC_ALIGN(name, ...) alignas(ALPAKA_OPTIMAL_ALIGNMENT(__VA_ARGS__)) __VA_ARGS__ name
#define PMACC_ALIGN8(name, ...) alignas(8) __VA_ARGS__ name

/*! area which is calculated
 *
 * CORE is the inner area of a grid
 * BORDER is the border of a grid (my own border, not the neighbor part)
 */
enum AreaType
{
    CORE = 1u, BORDER = 2u, GUARD = 4u
};

#define __delete(var) if((var)) { delete (var); var=NULL; }
#define __deleteArray(var) if((var)) { delete[] (var); var=NULL; }

} //namespace PMacc

/* Copyright 2016 Rene Widera
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

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace pmacc
{
    using IdxType = unsigned int;

    static constexpr uint32_t Dimensions = 3u;

    template<uint32_t T_dim>
    using AlpakaDim = ::alpaka::DimInt<T_dim>;
#if 0
    using KernelDim = AlpakaDim<Dimensions>;

    using IdxVec3 = ::alpaka::Vec<KernelDim, IdxType>;
#endif

    using AccHost = ::alpaka::DevCpu;
    using AccHostStream = ::alpaka::QueueCpuBlocking;

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)                   \
    || defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)                    \
    || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)

    using AccDev = ::alpaka::DevCpu;
#    if(CUPLA_STREAM_ASYNC_ENABLED == 1)
    using AccStream = ::alpaka::QueueCpuNonBlocking;
#    else
    using AccStream = ::alpaka::QueueCpuBlocking;
#    endif

#    ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
    using Acc = ::alpaka::AccCpuOmp2Threads<KernelDim, IdxType>;
#    endif

#    if(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED == 1)
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccCpuOmp2Blocks<AlpakaDim<T_dim>, IdxType>;
#    endif

#    ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
    using Acc = ::alpaka::AccCpuThreads<KernelDim, IdxType>;
#    endif

#    ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    using Acc = ::alpaka::AccCpuSerial<KernelDim, IdxType>;
#    endif

#    if(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED == 1)
    using Acc = ::alpaka::AccCpuTbbBlocks<KernelDim, IdxType>;
#    endif

#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using AccDev = ::alpaka::DevCudaRt;
#    if(CUPLA_STREAM_ASYNC_ENABLED == 1)
    using AccStream = ::alpaka::QueueCudaRtNonBlocking;
#    else
    using AccStream = ::alpaka::QueueCudaRtBlocking;
#    endif
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccGpuCudaRt<AlpakaDim<T_dim>, IdxType>;
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    using AccDev = ::alpaka::DevHipRt;
#    if(CUPLA_STREAM_ASYNC_ENABLED == 1)
    using AccStream = ::alpaka::QueueHipRtNonBlocking;
#    else
    using AccStream = ::alpaka::QueueHipRtBlocking;
#    endif
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccGpuHipRt<AlpakaDim<T_dim>, IdxType>;
#endif

    template<uint32_t T_dim>
    using AccThreadSeq = Acc<T_dim>;
    using AlpakaEventType = alpaka::Event<AccStream>;

    /*! device compile flag
     *
     * Enabled if the compiler processes currently a separate compile path for the device code
     *
     * @attention value is always 0 for alpaka CPU accelerators
     *
     * Value is 1 if device path is compiled else 0
     */
#if defined(__CUDA_ARCH__) || (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1 && defined(__HIP__))
#    define CUPLA_DEVICE_COMPILE 1
#else
#    define CUPLA_DEVICE_COMPILE 0
#endif

} // namespace pmacc

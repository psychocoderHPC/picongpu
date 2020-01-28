/*
  mallocMC: Memory Allocator for Many Core Architectures.
  http://www.icg.tugraz.at/project/mvp
  https://www.hzdr.de/crp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  Copyright (C) 2014 Institute of Radiation Physics,
                     Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
              Michael Kenzel - kenzel ( at ) icg.tugraz.at
              Carlchristian Eckert - c.eckert ( at ) hzdr.de

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

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <string>
#include <sstream>
#include <stdexcept>
#include <boost/cstdint.hpp>

#include "mallocMC_prefixes.hpp"

#include <hip/hip_runtime.h>

namespace CUDA
{
  class error : public std::runtime_error
  {
  private:
    static std::string genErrorString(hipError_t errorValue, const char* file, int line)
    {
      std::ostringstream msg;
      msg << file << '(' << line << "): error: " << hipGetErrorString(errorValue);
      return msg.str();
    }
  public:
    error(hipError_t errorValue, const char* file, int line)
      : runtime_error(genErrorString(errorValue, file, line))
    {
    }

    error(hipError_t errorValue)
      : runtime_error(hipGetErrorString(errorValue))
    {
    }

    error(const std::string& msg)
      : runtime_error(msg)
    {
    }
  };

  inline void checkError(hipError_t errorValue, const char* file, int line)
  {
    if (errorValue != hipSuccess)
      throw CUDA::error(errorValue, file, line);
  }

  inline void checkError(const char* file, int line)
  {
    checkError(hipGetLastError(), file, line);
  }

  inline void checkError()
  {
    hipError_t errorValue = hipGetLastError();
    if (errorValue != hipSuccess)
      throw CUDA::error(errorValue);
  }

#define MALLOCMC_CUDA_CHECKED_CALL(call) CUDA::checkError(call, __FILE__, __LINE__)
#define MALLOCMC_CUDA_CHECK_ERROR() CUDA::checkError(__FILE__, __LINE__)
}


namespace mallocMC
{

  template<int PSIZE>
  class __PointerEquivalent
  {
  public:
    typedef unsigned int type;
  };
  template<>
  class __PointerEquivalent<8>
  {
  public:
    typedef unsigned long long int type;
  };

  typedef mallocMC::__PointerEquivalent<sizeof(char*)>::type PointerEquivalent;

// https://github.com/ROCm-Developer-Tools/HIP/blob/master/include/hip/hcc_detail/device_functions.h
#ifdef __HIP__
  MAMC_ACCELERATOR inline boost::uint32_t laneid()
  {
    return __lane_id();
  }
#else
  MAMC_ACCELERATOR inline boost::uint32_t laneid()
  {
    boost::uint32_t mylaneid;
    asm("mov.u32 %0, %%laneid;" : "=r" (mylaneid));
    return mylaneid;
  }
#endif

  /** warp index within a multiprocessor
   *
   * Index of the warp within the multiprocessor at the moment of the query.
   * The result is volatile and can be different with each query.
   *
   * @return current index of the warp
   */
#ifdef __HIP__
  // get wave id https://github.com/ROCm-Developer-Tools/HIP/blob/f72a669487dd352e45321c4b3038f8fe2365c236/include/hip/hcc_detail/device_functions.h#L974-L1024
  __device__
  inline
  boost::uint32_t wave_id(void)
  {
      return __builtin_amdgcn_s_getreg(
              GETREG_IMMED(3, 0, 4));
  }
  MAMC_ACCELERATOR inline boost::uint32_t warpid()
  {
      return wave_id();
  }
#else
  MAMC_ACCELERATOR inline boost::uint32_t warpid()
  {
    boost::uint32_t mywarpid;
    asm("mov.u32 %0, %%warpid;" : "=r" (mywarpid));
    return mywarpid;
  }
#endif

#ifdef __HIP__
  MAMC_ACCELERATOR inline boost::uint32_t smid()
  {
    return __smid();
  }
#else
  MAMC_ACCELERATOR inline boost::uint32_t smid()
  {
    boost::uint32_t mysmid;
    asm("mov.u32 %0, %%smid;" : "=r" (mysmid));
    return mysmid;
  }
#endif

#ifdef __HIP__
  MAMC_ACCELERATOR inline boost::uint64_t lanemask_lt()
  {
      return __lanemask_lt();
  }
#else
  MAMC_ACCELERATOR inline boost::uint32_t lanemask_lt()
  {
    boost::uint32_t lanemask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r" (lanemask));
    return lanemask;
  }
#endif


  template<class T>
  MAMC_HOST MAMC_ACCELERATOR inline T divup(T a, T b) { return (a + b - 1)/b; }

  /** the maximal number threads per block
   *
   * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
   */
  struct MaxThreadsPerBlock
  {
    // valid for sm_2.X - sm_7.5
    BOOST_STATIC_CONSTEXPR uint32_t value = 1024;
  };

  /** number of threads within a warp
   *
   * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
   */
  struct WarpSize
  {
#if __CUDA_ARCH__
    // valid for sm_2.X - sm_7.5
    BOOST_STATIC_CONSTEXPR uint32_t value = 32;
#else
    BOOST_STATIC_CONSTEXPR uint32_t value = 64;
#endif
  };

  /** warp id within a cuda block
   *
   * The id is constant over the lifetime of the thread.
   * The id is not equal to warpid().
   *
   * @return warp id within the block
   */
  MAMC_ACCELERATOR inline boost::uint32_t warpid_withinblock()
  {
#if __CUDA_ARCH__
    return (
      threadIdx.z * blockDim.y * blockDim.x +
      threadIdx.y * blockDim.x +
      threadIdx.x
    ) / WarpSize::value;
#else
    return (
      hipThreadIdx_z * hipBlockDim_y * hipBlockDim_x +
      hipThreadIdx_y * hipBlockDim_x +
      hipThreadIdx_x
    ) / WarpSize::value;
#endif
  }
}

/*
  mallocMC: Memory Allocator for Many Core Architectures.

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

#pragma once

#include <string>

#include "../mallocMC_utils.hpp"

#include "SimpleCudaMalloc.hpp"

namespace mallocMC{
namespace ReservePoolPolicies{

  struct SimpleCudaMalloc{
    static void* setMemPool(size_t memsize){
      void* pool = NULL;
      MALLOCMC_CUDA_CHECKED_CALL(cudaMalloc(&pool, memsize));
      return pool;
    }

    static void resetMemPool(void* p){
        if(p != nullptr)
        {
            MALLOCMC_CUDA_CHECKED_CALL(cudaDeviceSynchronize());
            MALLOCMC_CUDA_CHECKED_CALL(cudaGetLastError());
            MALLOCMC_CUDA_CHECKED_CALL(cudaFree(p));
        }
    }

    static std::string classname(){
      return "SimpleCudaMalloc";
    }

  };

} //namespace ReservePoolPolicies
} //namespace mallocMC

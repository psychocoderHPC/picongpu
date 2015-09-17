/*
  mallocMC: Memory Allocator for Many Core Architectures.
  http://www.icg.tugraz.at/project/mvp
  https://www.hzdr.de/crp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  Copyright (C) 2014 Institute of Radiation Physics,
                     Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
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

#include "mallocMC_prefixes.hpp"
#include <vector>

/** Creates a global typedef
 *
 * Will create a global typedef of the supplied type. This should be done first.
 */
#define MALLOCMC_TYPEDEF(MALLOCMC_USER_DEFINED_TYPENAME)                       \
namespace mallocMC{                                                            \
    typedef MALLOCMC_USER_DEFINED_TYPENAME mallocMCType;                       \
}

/** Creates a global object of a many core memory allocator
 *
 * Will create a global object. This should be done after defining a new 
 * many core memory allocator with a typedef.
 */
#if defined(MAMC_CUDA_ENABLED) && defined(__CUDACC__)
    #define MALLOCMC_OBJECT()                                                  \
        namespace mallocMC{                                                    \
            MAMC_ACC_CUDA_ONLY mallocMCType mallocMCGlobalObject;      \
        }
#else
    #define MALLOCMC_OBJECT()                                                  \
        namespace mallocMC{                                                    \
            mallocMCType mallocMCGlobalObject;                                 \
        }
#endif

/** Creates a global object of a many core memory allocator
 *
 * Will create global heap functions that use the allocator. This should be done after
 * defining a new many core memory allocator with a typedef.
 */
#define MALLOCMC_HEAP()                                                        \
namespace mallocMC{                                                            \
MAMC_HOST void* initHeap(                                                      \
    size_t heapsize = 8U*1024U*1024U,                                          \
    mallocMCType &p = mallocMCGlobalObject                                     \
    )                                                                          \
{                                                                              \
    return p.initHeap(heapsize);                                               \
}                                                                              \
MAMC_HOST void finalizeHeap(                                                   \
    mallocMCType &p = mallocMCGlobalObject                                     \
    )                                                                          \
{                                                                              \
    p.finalizeHeap();                                                          \
}                                                                              \
} /* end namespace mallocMC */


/** Provides the easily accessible functions getAvaliableSlots
 *
 * Will use the global object defined by MALLOCMC_SET_ALLOCATOR_TYPE and
 * use it to generate global functions that use this type internally.
 */
#define MALLOCMC_AVAILABLESLOTS()                                              \
namespace mallocMC{                                                            \
_Pragma("hd_warning_disable")                                                  \
MAMC_ACC                                                               \
unsigned getAvailableSlots(                                                    \
    size_t slotSize,                                                           \
    mallocMCType &p = mallocMCGlobalObject)                                    \
{                                                                              \
    return p.getAvailableSlots(slotSize);                                      \
}                                                                              \
_Pragma("hd_warning_disable")                                                  \
MAMC_ACC                                                               \
bool providesAvailableSlots()                                                  \
{                                                                              \
    return Traits<mallocMCType>::providesAvailableSlots;                       \
}                                                                              \
} /* end namespace mallocMC */

/** __THROW is defined in Glibc so it is not available on all platforms.
 */
#ifndef __THROW
  #define __THROW
#endif

/** Create the functions malloc() and free() inside a namespace
 *
 * This allows for a peaceful coexistence between different functions called
 * "malloc" or "free". This is useful when using a policy that contains a call
 * to the original device-side malloc() from CUDA.
 */
#define MALLOCMC_MALLOC()                                                      \
namespace mallocMC{                                                            \
_Pragma("hd_warning_disable")                                                  \
MAMC_ACC                                                               \
void* malloc(size_t t) __THROW                                                 \
{                                                                              \
  return mallocMC::mallocMCGlobalObject.alloc(t);                              \
}                                                                              \
_Pragma("hd_warning_disable")                                                  \
MAMC_ACC                                                               \
void free(void* p) __THROW                                                     \
{                                                                              \
  mallocMC::mallocMCGlobalObject.dealloc(p);                                   \
}                                                                              \
} /* end namespace mallocMC */

/** Create the function getHeapLocations inside a namespace
 *
 * This returns a vector of type mallocMC::HeapInfo. The HeapInfo should at least
 * contain the pointer to the heap (on device) and its size.
 */
#define MALLOCMC_HEAPLOC()                                                     \
namespace mallocMC{                                                            \
MAMC_HOST                                                                      \
std::vector<mallocMC::HeapInfo> getHeapLocations()                             \
{                                                                              \
  return mallocMC::mallocMCGlobalObject.getHeapLocations();                    \
}                                                                              \
} /* end namespace mallocMC */

/** Set up the global variables and functions
 *
 * This Macro should be called with the type of a newly defined allocator. It
 * will create a global object of that allocator and the necessary functions to
 * initialize and allocate memory.
 */
#define MALLOCMC_SET_ALLOCATOR_TYPE(MALLOCMC_USER_DEFINED_TYPE)                \
MALLOCMC_TYPEDEF(MALLOCMC_USER_DEFINED_TYPE)                                   \
MALLOCMC_OBJECT()                                                              \
MALLOCMC_HEAP()                                                                \
MALLOCMC_MALLOC()                                                              \
MALLOCMC_HEAPLOC()                                                             \
MALLOCMC_AVAILABLESLOTS()

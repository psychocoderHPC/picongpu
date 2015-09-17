/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2015 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Benjamin Worpitz - HZDR

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

#include <boost/cstdint.hpp>
#include <boost/mpl/bool.hpp>
#include <iostream>

#include "HostNew.hpp"

namespace mallocMC
{
namespace CreationPolicies
{

class HostNew
{
    typedef boost::uint32_t uint32;
    typedef boost::uint8_t uint8;

public:
    typedef boost::mpl::bool_<false> providesAvailableSlots;

    MAMC_HOST
    void* create(uint32 bytes)
    {
      return reinterpret_cast<void*>(new uint8[bytes]);
    }

    MAMC_HOST
    void destroy(void* mem)
    {
        delete[] reinterpret_cast<uint8*> (mem);
    }

    MAMC_HOST
    bool isOOM( void* p, size_t s )
    {
        return s && (p == NULL);
    }

    template < typename T>
    MAMC_HOST
    static void* initHeap( const T& obj, void* pool, size_t memsize )
    {
        return const_cast<void *> (reinterpret_cast<void const *> (&obj));
    }

    template < typename T>
    MAMC_HOST
    static void finalizeHeap( const T& obj, void* pool )
    {
        return;
    }

    MAMC_HOST
    static std::string classname( )
    {
        return "HostNew";
    }

};

} //namespace CreationPolicies
} //namespace mallocMC

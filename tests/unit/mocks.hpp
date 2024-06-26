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

#include <alpaka/atomic/AtomicAtomicRef.hpp>
#include <alpaka/intrinsic/IntrinsicFallback.hpp>
#include <alpaka/mem/fence/Traits.hpp>

// TODO(lenz): This is a dirty hack. I'm using AtomicAtomicRef instead of an accelerator directly because it turns out
// that it's very hard to instantiate an accelerator and the atomics don't really care what you hand them. But the
// mem_fence DOES care, so I have to provide this empty implementation. We don't really get a thread fence anymore, of
// course, but that's okay because we are single-threaded in this file. Never do this in production, of course!
template<>
struct alpaka::trait::MemFence<alpaka::AtomicAtomicRef, alpaka::memory_scope::Device, void>
{
    template<typename... T>
    ALPAKA_FN_ACC static void mem_fence(T... /*We're just providing a general interface.*/)
    {
    }
};

template<>
struct alpaka::trait::Ffs<alpaka::AtomicAtomicRef, void> : alpaka::trait::Ffs<alpaka::IntrinsicFallback, void>
{
    template<typename T, typename... TArgs>
    static auto ffs(T /*intrinsic*/, TArgs... args)
    {
        return Ffs<alpaka::IntrinsicFallback, void>::ffs(alpaka::IntrinsicFallback{}, args...);
    }
};

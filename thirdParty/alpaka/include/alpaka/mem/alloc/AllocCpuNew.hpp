/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <alpaka/mem/alloc/Traits.hpp>

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>
#include <iostream>
#include <alpaka/core/Cuda.hpp>

namespace alpaka
{
    namespace mem
    {
        //-----------------------------------------------------------------------------
        //! The allocator specifics.
        namespace alloc
        {
            //#############################################################################
            //! The CPU new allocator.
            class AllocCpuNew
            {
            public:
                using AllocBase = AllocCpuNew;
            };

            namespace traits
            {
                //#############################################################################
                //! The CPU new allocator memory allocation trait specialization.
                template<
                    typename T>
                struct Alloc<
                    T,
                    AllocCpuNew>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto alloc(
                        AllocCpuNew const & alloc,
                        std::size_t const & sizeElems)
                    -> T *
                    {
                        alpaka::ignore_unused(alloc);
                        auto tmp = new T[sizeElems];
                       // std::cerr<<"alloc[] mem: "<<(int*)tmp<<std::endl;
                        return tmp;
                    }
                };

                //#############################################################################
                //! The CPU new allocator memory free trait specialization.
                template<
                    typename T>
                struct Free<
                    T,
                    AllocCpuNew>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto free(
                        AllocCpuNew const & alloc,
                        T const * const ptr)
                    -> void
                    {
                       // std::cerr<<"free[] mem: "<<(int*)ptr<<std::endl;
                        alpaka::ignore_unused(alloc);
                        return delete[] ptr;
                    }
                };
            }
        }
    }
}

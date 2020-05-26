/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#if _OPENACC < 201306
    #error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC xx or higher!
#endif

#include <type_traits>
#include <array>

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace dyn
            {
                //#############################################################################
                //! The OpenACC block shared memory allocator.
                class BlockSharedMemDynOacc
                {
                public:
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynOacc() = default;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynOacc(BlockSharedMemDynOacc const &) = delete;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynOacc(BlockSharedMemDynOacc &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynOacc const &) -> BlockSharedMemDynOacc & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynOacc &&) -> BlockSharedMemDynOacc & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemDynOacc() = default;

                    class BlockShared
                    {
                        mutable std::array<std::size_t, 30<<(10-sizeof(std::size_t))> m_mem; // ! static 30kB
                        std::size_t m_dynSize;

                    public:
                        //-----------------------------------------------------------------------------
                        BlockShared(std::size_t sizeBytes) : m_dynSize(sizeBytes) {}
                        //-----------------------------------------------------------------------------
                        BlockShared(BlockShared const &) = delete;
                        //-----------------------------------------------------------------------------
                        BlockShared(BlockShared &&) = delete;
                        //-----------------------------------------------------------------------------
                        auto operator=(BlockShared const &) -> BlockShared & = delete;
                        //-----------------------------------------------------------------------------
                        auto operator=(BlockShared &&) -> BlockShared & = delete;
                        //-----------------------------------------------------------------------------
                        /*virtual*/ ~BlockShared() = default;

                        std::size_t* dynMemBegin() const {return m_mem.data();}
                        std::size_t* staticMemBegin() const {return m_mem.data()+m_dynSize;}
                    };
                };
            }
        }
    }
}

#endif

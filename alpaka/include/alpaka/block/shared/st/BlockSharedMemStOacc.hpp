/* Copyright 2019 Benjamin Worpitz, Erik Zenker, René Widera
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

#include <alpaka/block/shared/st/Traits.hpp>
#include <alpaka/block/sync/Traits.hpp>

#include <type_traits>
#include <cstdint>

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace st
            {
                //#############################################################################
                //! The GPU CUDA block shared memory allocator.
                class BlockSharedMemStOacc
                {
                public:
                    //-----------------------------------------------------------------------------
                    BlockSharedMemStOacc() = default;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemStOacc(BlockSharedMemStOacc const &) = delete;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemStOacc(BlockSharedMemStOacc &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemStOacc const &) -> BlockSharedMemStOacc & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemStOacc &&) -> BlockSharedMemStOacc & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemStOacc() = default;

                    class BlockShared
                    {
                        mutable unsigned int m_allocdWords = 0;
                        mutable size_t* m_mem;
                        template<typename T>
                        static constexpr size_t alignPitch()
                        {
                            return sizeof(T)/sizeof(size_t) + (sizeof(T)/sizeof(size_t)>0);
                        }

                        public:

                        BlockShared(size_t* mem) : m_mem(mem) {}
                        //-----------------------------------------------------------------------------
                        BlockShared(BlockSharedMemStOacc const &) = delete;
                        //-----------------------------------------------------------------------------
                        BlockShared(BlockSharedMemStOacc &&) = delete;
                        //-----------------------------------------------------------------------------
                        auto operator=(BlockShared const &) -> BlockShared & = delete;
                        //-----------------------------------------------------------------------------
                        auto operator=(BlockShared &&) -> BlockShared & = delete;
                        //-----------------------------------------------------------------------------
                        /*virtual*/ ~BlockShared() = default;

                        template <typename T>
                        void alloc() const
                        {
                            size_t* buf = &m_mem[m_allocdWords];
                            new (buf) T();
                            m_allocdWords += alignPitch<T>();
                        }

                        template <typename T>
                        T& getLatestVar() const
                        {
                           return *reinterpret_cast<T*>(&m_mem[m_allocdWords-alignPitch<T>()]);
                        }
                    };
                };
            }
        }
    }
}

#endif

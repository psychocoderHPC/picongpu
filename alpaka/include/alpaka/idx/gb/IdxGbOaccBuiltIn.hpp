/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
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

#include <alpaka/idx/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/idx/MapIdx.hpp>

namespace alpaka
{
    namespace idx
    {
        namespace gb
        {
            //#############################################################################
            //! The CUDA accelerator ND index provider.
            template<
                typename TDim,
                typename TIdx>
            class IdxGbOaccBuiltIn
            {
            public:
                //-----------------------------------------------------------------------------
                IdxGbOaccBuiltIn() = default;
                //-----------------------------------------------------------------------------
                IdxGbOaccBuiltIn(IdxGbOaccBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                IdxGbOaccBuiltIn(IdxGbOaccBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxGbOaccBuiltIn const & ) -> IdxGbOaccBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxGbOaccBuiltIn &&) -> IdxGbOaccBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxGbOaccBuiltIn() = default;

                class BlockShared : public concepts::Implements<ConceptIdxGb, BlockShared>
                {
                public:
                    //-----------------------------------------------------------------------------
                    BlockShared(const TIdx &gridBlockIdx) : m_gridBlockIdx(gridBlockIdx) {}
                    //-----------------------------------------------------------------------------
                    BlockShared(BlockShared const &) = delete;
                    //-----------------------------------------------------------------------------
                    BlockShared(BlockShared &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockShared const & ) -> BlockShared & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockShared &&) -> BlockShared & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockShared() = default;

                    TIdx const m_gridBlockIdx;
                };
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator index dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                idx::gb::IdxGbOaccBuiltIn<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }

    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator grid block index idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::gb::IdxGbOaccBuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif

/**
 * Copyright 2017 Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc_types.hpp"
#include <type_traits>

namespace PMacc
{
namespace mappings
{
namespace threads
{

    template<
        uint32_t T_domainDim,
        uint32_t T_workerDim,
        uint32_t T_simdDim = 1
    >
    struct ForEachIdx
    {
        static constexpr uint32_t domainDim = T_domainDim;
        static constexpr uint32_t workerDim = T_workerDim;
        static constexpr uint32_t simdDim = T_simdDim;

        uint32_t const m_workerIdx;

        static constexpr uint32_t numCollectiveIterations =
            (( domainDim + simdDim * workerDim - 1 ) / ( simdDim * workerDim));

        static constexpr bool outerLoopCondition =
            ( domainDim % (simdDim * workerDim) ) == 0 ||
            ( simdDim * workerDim == 1 );

        static constexpr bool innerLoopCondition =
            ( domainDim % simdDim ) == 0 ||
            ( simdDim == 1 );

        HDINLINE ForEachIdx( uint32_t const workerIdx ) : m_workerIdx( workerIdx )
        {
        }

        template<
            typename T_Functor,
            typename ... T_Args
        >
        HDINLINE void
        operator()(
            T_Functor const & functor,
            T_Args && ... args
        ) const
        {
            for( uint32_t i = 0; i < numCollectiveIterations; ++i )
            {
                const uint32_t beginWorker = i * simdDim;
                const uint32_t beginIdx = beginWorker * workerDim + simdDim * m_workerIdx;
                if(
                    outerLoopCondition ||
                    !innerLoopCondition ||
                    beginIdx < domainDim
                )
                {
                    for( uint32_t j = 0; j < simdDim; ++j )
                    {
                        const uint32_t localIdx = beginIdx + j;
                        if(
                            innerLoopCondition ||
                            localIdx < domainDim
                        )
                            functor(
                                localIdx,
                                beginWorker + j,
                                std::forward(args) ...
                            );
                    }
                }
            }
        }

        template<
            typename T_Functor,
            typename ... T_Args
        >
        HDINLINE void
        operator()(
            T_Functor & functor,
            T_Args && ... args
        ) const
        {
            for( uint32_t i = 0; i < numCollectiveIterations; ++i )
            {
                const uint32_t beginWorker = i * simdDim;
                const uint32_t beginIdx = beginWorker * workerDim + simdDim * m_workerIdx;
                if(
                    outerLoopCondition ||
                    !innerLoopCondition ||
                    beginIdx < domainDim
                )
                {
                    for( uint32_t j = 0; j < simdDim; ++j )
                    {
                        const uint32_t localIdx = beginIdx + j;
                        if(
                            innerLoopCondition ||
                            localIdx < domainDim
                        )
                            functor(
                                localIdx,
                                beginWorker + j,
                                std::forward(args) ...
                            );
                    }
                }
            }
        }

    };

} // namespace threads
} // namespace mappings
} // namespace PMacc

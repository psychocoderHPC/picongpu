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

#include "mappings/threads/IdxConfig.hpp"
#include "pmacc_types.hpp"

namespace PMacc
{
namespace mappings
{
namespace threads
{

    template<
        typename T_IdxConfig
    >
    struct ForEachIdx : public T_IdxConfig
    {
        using T_IdxConfig::domainSize;
        using T_IdxConfig::workerSize;
        using T_IdxConfig::simdSize;
        using T_IdxConfig::collectiveOps;

        uint32_t const m_workerIdx;

        static constexpr bool outerLoopCondition =
            ( domainSize % (simdSize * workerSize) ) == 0 ||
            ( simdSize * workerSize == 1 );

        static constexpr bool innerLoopCondition =
            ( domainSize % simdSize ) == 0 ||
            ( simdSize == 1 );

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
            for( uint32_t i = 0; i < collectiveOps; ++i )
            {
                const uint32_t beginWorker = i * simdSize;
                const uint32_t beginIdx = beginWorker * workerSize + simdSize * m_workerIdx;
                if(
                    outerLoopCondition ||
                    !innerLoopCondition ||
                    beginIdx < domainSize
                )
                {
                    for( uint32_t j = 0; j < simdSize; ++j )
                    {
                        const uint32_t localIdx = beginIdx + j;
                        if(
                            innerLoopCondition ||
                            localIdx < domainSize
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
            for( uint32_t i = 0; i < collectiveOps; ++i )
            {
                const uint32_t beginWorker = i * simdSize;
                const uint32_t beginIdx = beginWorker * workerSize + simdSize * m_workerIdx;
                if(
                    outerLoopCondition ||
                    !innerLoopCondition ||
                    beginIdx < domainSize
                )
                {
                    for( uint32_t j = 0; j < simdSize; ++j )
                    {
                        const uint32_t localIdx = beginIdx + j;
                        if(
                            innerLoopCondition ||
                            localIdx < domainSize
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

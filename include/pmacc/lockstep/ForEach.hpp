/* Copyright 2017-2021 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/lockstep/Config.hpp"
#include "pmacc/lockstep/Idx.hpp"
#include "pmacc/lockstep/Worker.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace lockstep
    {
        /** describe a constant index domain
         *
         * describe the size of the index domain and the number of workers to operate on the domain
         *
         * @tparam T_domainSize number of indices in the domain
         * @tparam T_numWorkers number of worker working on @p T_domainSize
         * @tparam T_simdSize SIMD width
         */
        template<typename T_Config>
        struct ForEach
            : T_Config
            , Worker<T_Config::numWorkers>
        {
            using BaseConfig = T_Config;

            using BaseConfig::domainSize;
            using BaseConfig::numCollIter;
            using BaseConfig::numWorkers;
            using BaseConfig::simdSize;

            /** constructor
             *
             * @param workerIdx index of the worker: range [0;workerSize)
             */
            HDINLINE
            ForEach(uint32_t const workerIdx) : Worker<numWorkers>(workerIdx)
            {
            }

            HDINLINE Worker<numWorkers> getWorkerCfg() const
            {
                return static_cast<Worker<numWorkers>>(*this);
            }

            /** execute a functor
             *
             * Distribute the indices even over all worker and execute a user defined functor.
             * There is no guarantee in which order the indices will be processed.
             *
             * @param functor is called for each index which is mapped to the worker
             * @param args optional arguments forwarded to the functor
             *
             * @{
             */

            /** The functor must fulfill the following interface:
             * @code
             * template< uint32_t T_domainSize >
             * void operator()( lockstep::Idx< T_domainSize > const idx );
             * // or
             * template< uint32_t T_domainSize>
             * void operator()( uint32_t const linearIdx );
             * @endcode
             */
            template<typename T_Functor>
            HDINLINE auto operator()(T_Functor&& functor) const -> decltype(functor(std::declval<Idx const>()))
            {
                for(uint32_t i = 0u; i < numCollIter; ++i)
                {
                    uint32_t const beginWorker = i * simdSize;
                    uint32_t const beginIdx = beginWorker * numWorkers + simdSize * this->getWorkerIdx();
                    if(outerLoopCondition || !innerLoopCondition || beginIdx < domainSize)
                    {
                        for(uint32_t j = 0u; j < simdSize; ++j)
                        {
                            uint32_t const localIdx = beginIdx + j;
                            if(innerLoopCondition || localIdx < domainSize)
                                functor(Idx(localIdx, beginWorker + j));
                        }
                    }
                }
            }

            /** The functor must fulfill the following interface:
             * @code
             * template< uint32_t T_domainSize, typename ... T_Args >
             * void operator()( T_Args && ... );
             * @endcode
             */
            template<typename T_Functor>
            HDINLINE auto operator()(T_Functor&& functor) const -> decltype(functor())
            {
                for(uint32_t i = 0u; i < numCollIter; ++i)
                {
                    uint32_t const beginWorker = i * simdSize;
                    uint32_t const beginIdx = beginWorker * numWorkers + simdSize * this->getWorkerIdx();
                    if(outerLoopCondition || !innerLoopCondition || beginIdx < domainSize)
                    {
                        for(uint32_t j = 0u; j < simdSize; ++j)
                        {
                            uint32_t const localIdx = beginIdx + j;
                            if(innerLoopCondition || localIdx < domainSize)
                                functor();
                        }
                    }
                }
            }

            /** @} */

        private:
            static constexpr bool outerLoopCondition
                = (domainSize % (simdSize * numWorkers)) == 0u || (simdSize * numWorkers == 1u);

            static constexpr bool innerLoopCondition = (domainSize % simdSize) == 0u || (simdSize == 1u);
        };

        template<uint32_t T_numWorkers>
        using Master = ForEach<Config<1, T_numWorkers, 1>>;

        template<uint32_t T_domainSize, uint32_t T_numWorkers, uint32_t T_simdSize = 1>
        HDINLINE auto makeForEach(uint32_t const workerIdx)
        {
            return ForEach<Config<T_domainSize, T_numWorkers, T_simdSize>>(workerIdx);
        }

        template<uint32_t T_numWorkers>
        HDINLINE auto makeMaster(uint32_t const workerIdx)
        {
            return Master<T_numWorkers>(workerIdx);
        }

    } // namespace lockstep
} // namespace pmacc

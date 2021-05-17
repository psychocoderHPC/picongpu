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

#include "pmacc/mappings/threads/DomainIdx.hpp"
#include "pmacc/mappings/threads/ForEachIdx.hpp"
#include "pmacc/mappings/threads/IdxConfig.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/types.hpp"

#include <type_traits>
#include <utility>

namespace pmacc
{
    namespace memory
    {
        /** Variable used by virtual worker
         *
         * This object is designed to hold context variables in lock step
         * programming. A context variable is just a local variable of a virtual
         * worker. Allocating and using a context variable allows to propagate
         * virtual worker states over subsequent lock steps. A context variable
         * for a set of virtual workers is owned by their (physical) worker.
         *
         * Data stored in a context variable should only be used with a lock step
         * programming construct e.g. ForEachIdx<>
         */
        template<typename T_Type, typename T_IdxConfig>
        struct CtxVar
            : protected Array<T_Type, T_IdxConfig::numCollIter * T_IdxConfig::simdSize>
            , T_IdxConfig
        {
            using T_IdxConfig::domainSize;
            using T_IdxConfig::numCollIter;
            using T_IdxConfig::simdSize;
            using T_IdxConfig::workerSize;

            using BaseArray = Array<T_Type, T_IdxConfig::numCollIter * T_IdxConfig::simdSize>;

            /** default constructor
             *
             * Data member are uninitialized.
             * This method must be called collectively by all workers.
             */
            CtxVar() = default;

            /** constructor
             *
             * Initialize each member with the given value.
             * This method must be called collectively by all workers.
             *
             * @param value element assigned to each member
             */
            HDINLINE explicit CtxVar(T_Type const& value) : BaseArray(value)
            {
            }

            /** disable copy constructor
             */
            HDINLINE CtxVar(CtxVar const&) = delete;

            /** constructor
             *
             * Initialize each member with the result of the given functor.
             * This method must be called collectively by all workers.
             *
             * @tparam T_Functor type of the user functor
             * @tparam T_Args type of user parameters
             * @param w of worker range: [0;workerSize)
             * @param functor functor to initialize the member ( need to implement `::operator(size_type idx)`)
             * @param args user defined arguments those should forwarded to the functor
             */
            template<typename T_Functor, typename... T_Args>
            HDINLINE explicit CtxVar(uint32_t const workerIdx, T_Functor&& functor, T_Args&&... args)
            {
                initData(workerIdx, std::forward<T_Functor>(functor), std::forward<T_Args>(args)...);
            }

            /** get element for the worker
             *
             * @tparam T_Idx any type which can be implicit casted to an integral type
             * @param idx index within the array
             *
             * @{
             */
            HDINLINE typename BaseArray::const_reference operator[](mappings::threads::DomainIdx const domIdx) const
            {
                return reinterpret_cast<T_Type const*>(BaseArray::data())[domIdx.workerElemIdx];
            }

            HDINLINE typename BaseArray::reference operator[](mappings::threads::DomainIdx const domIdx)
            {
                return reinterpret_cast<T_Type*>(BaseArray::data())[domIdx.workerElemIdx];
            }
            /** @} */

        private:
            /** initialize the context variable
             *
             * @param workerIdx index of worker range: [0;workerSize)
             * @param functor functor to initialize the member ( need to implement `::operator(size_type idx)`)
             * @param args user defined arguments those should forwarded to the functor
             *
             * @{
             */

            /** The functor must fulfill the following interface:
             * @code
             * template< uint32_t T_domainSize, typename ... T_Args >
             * auto operator()( DomainIdx< T_domainSize > const domIdx, T_Args && ... );
             * @endcode
             */
            template<typename T_Functor, typename... T_Args>
            HDINLINE auto initData(uint32_t const workerIdx, T_Functor&& functor, T_Args&&... args)
                -> std::enable_if_t<
                    sizeof(decltype(
                        functor(std::declval<mappings::threads::DomainIdx const>(), std::forward<T_Args>(args)...)))
                    != 0>
            {
                mappings::threads::ForEachIdx<T_IdxConfig>{workerIdx}(
                    [&, this](mappings::threads::DomainIdx const domIdx) {
                        (*this)[domIdx] = functor(domIdx, std::forward<T_Args>(args)...);
                    });
            }

            /** The functor must fulfill the following interface:
             * @code
             * template< uint32_t T_domainSize, typename ... T_Args >
             * auto operator()( uint32_t const linearIdx, T_Args && ... );
             * @endcode
             */
            template<typename T_Functor, typename... T_Args>
            HDINLINE auto initData(uint32_t const workerIdx, T_Functor&& functor, T_Args&&... args)
                -> std::enable_if_t<
                    sizeof(decltype(functor(std::declval<uint32_t const>(), std::forward<T_Args>(args)...))) != 0>
            {
                mappings::threads::ForEachIdx<T_IdxConfig>{workerIdx}(
                    [&, this](mappings::threads::DomainIdx const domIdx) {
                        (*this)[domIdx] = functor(domIdx.lIdx(), std::forward<T_Args>(args)...);
                    });
            }

            /** The functor must fulfill the following interface:
             * @code
             * template< uint32_t T_domainSize, typename ... T_Args >
             * auto operator()( T_Args && ... );
             * @endcode
             */
            template<typename T_Functor, typename... T_Args>
            HDINLINE auto initData(uint32_t const workerIdx, T_Functor&& functor, T_Args&&... args)
                -> std::enable_if_t<sizeof(decltype(functor(std::forward<T_Args>(args)...))) != 0>
            {
                mappings::threads::ForEachIdx<T_IdxConfig>{workerIdx}(
                    [&, this](mappings::threads::DomainIdx const domIdx) {
                        (*this)[domIdx] = functor(std::forward<T_Args>(args)...);
                    });
            }

            /** @} */
        };

    } // namespace memory
} // namespace pmacc

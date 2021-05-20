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
#include "pmacc/lockstep/ForEach.hpp"
#include "pmacc/lockstep/Idx.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/types.hpp"

#include <type_traits>
#include <utility>

namespace pmacc
{
    namespace lockstep
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
        template<typename T_Type, typename T_Config>
        struct Variable
            : protected memory::Array<T_Type, T_Config::numCollIter * T_Config::simdSize>
            , T_Config
        {
            using T_Config::domainSize;
            using T_Config::numCollIter;
            using T_Config::numWorkers;
            using T_Config::simdSize;

            using BaseArray = memory::Array<T_Type, T_Config::numCollIter * T_Config::simdSize>;

            /** default constructor
             *
             * Data member are uninitialized.
             * This method must be called collectively by all workers.
             */
            Variable() = default;

            /** constructor
             *
             * Initialize each member with the given value.
             * This method must be called collectively by all workers.
             *
             * @param value element assigned to each member
             */
            HDINLINE explicit Variable(T_Type const& value) : BaseArray(value)
            {
            }

            /** disable copy constructor
             */
            HDINLINE Variable(Variable const&) = delete;

            HDINLINE Variable(Variable&&) = default;

            HDINLINE Variable& operator=(Variable&&) = default;

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
            template<typename T_Functor>
            HDINLINE explicit Variable(uint32_t const& workerIdx, T_Functor&& functor)
            {
                initData(workerIdx, std::forward<T_Functor>(functor));
            }

            /** get element for the worker
             *
             * @tparam T_Idx any type which can be implicit casted to an integral type
             * @param idx index within the array
             *
             * @{
             */
            HDINLINE typename BaseArray::const_reference operator[](Idx const idx) const
            {
                return reinterpret_cast<T_Type const*>(BaseArray::data())[idx.workerElemIdx];
            }

            HDINLINE typename BaseArray::reference operator[](Idx const idx)
            {
                return reinterpret_cast<T_Type*>(BaseArray::data())[idx.workerElemIdx];
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
             * auto operator()( lockstep::Idx< T_domainSize > const idx );
             * // or
             * template< uint32_t T_domainSize >
             * auto operator()( uint32_t const linearIdx );
             * @endcode
             */
            template<typename T_Functor>
            HDINLINE auto initData(uint32_t const workerIdx, T_Functor&& functor)
                -> std::enable_if_t<sizeof(decltype(functor(std::declval<lockstep::Idx const>()))) != 0>
            {
                ForEach<T_Config>{workerIdx}([&, this](Idx idx) { (*this)[idx] = functor(idx); });
            }

            /** The functor must fulfill the following interface:
             * @code
             * template< uint32_t T_domainSize, typename ... T_Args >
             * auto operator()( T_Args && ... );
             * @endcode
             */
            template<typename T_Functor>
            HDINLINE auto initData(uint32_t const workerIdx, T_Functor&& functor)
                -> std::enable_if_t<sizeof(decltype(functor())) != 0>
            {
                ForEach<T_Config>{workerIdx}([&, this](lockstep::Idx const idx) { (*this)[idx] = functor(); });
            }

            /** @} */
        };


        template<typename T_Type, uint32_t T_domainSize, uint32_t T_numWorkers, uint32_t T_simdSize>
        HDINLINE auto makeVar(ForEach<Config<T_domainSize, T_numWorkers, T_simdSize>> const& forEach)
        {
            return Variable<T_Type, typename ForEach<Config<T_domainSize, T_numWorkers, T_simdSize>>::BaseConfig>();
        }

        template<typename T_Type, uint32_t T_domainSize, uint32_t T_numWorkers, uint32_t T_simdSize>
        HDINLINE auto makeVar(
            ForEach<Config<T_domainSize, T_numWorkers, T_simdSize>> const& forEach,
            T_Type const& value)
        {
            return Variable<T_Type, typename ForEach<Config<T_domainSize, T_numWorkers, T_simdSize>>::BaseConfig>(
                value);
        }

        template<
            typename T_Type,
            uint32_t T_domainSize,
            uint32_t T_numWorkers,
            uint32_t T_simdSize,
            typename T_Functor>
        HDINLINE auto makeVar(
            T_Functor&& functor,
            ForEach<Config<T_domainSize, T_numWorkers, T_simdSize>> const& forEach)
        {
            return Variable<T_Type, typename ForEach<Config<T_domainSize, T_numWorkers, T_simdSize>>::BaseConfig>(
                forEach.getWorkerIdx(),
                std::forward<T_Functor>(functor));
        }

    } // namespace lockstep
} // namespace pmacc

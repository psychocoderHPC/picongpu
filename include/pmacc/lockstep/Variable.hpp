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
#include "pmacc/memory/Array.hpp"
#include "pmacc/types.hpp"

#include <type_traits>
#include <utility>

namespace pmacc
{
    namespace lockstep
    {
        template<typename T_Config>
        struct ForEach;

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
        struct alignas(alignof(T_Type)) Variable
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
            template<typename... T_Args>
            HDINLINE explicit Variable(T_Args&&... args) : BaseArray(std::forward<T_Args>(args)...)
            {
            }

            /** disable copy constructor
             */
            HDINLINE Variable(Variable const&) = delete;

            HDINLINE Variable(Variable&&) = default;

            HDINLINE Variable& operator=(Variable&&) = default;

            /** get element for the worker
             *
             * @tparam T_Idx any type which can be implicit casted to an integral type
             * @param idx index within the array
             *
             * @{
             */
            HDINLINE typename BaseArray::const_reference operator[](Idx const idx) const
            {
                return BaseArray::operator[](idx.workerElemIdx);
            }

            HDINLINE typename BaseArray::reference operator[](Idx const idx)
            {
                return BaseArray::operator[](idx.workerElemIdx);
            }
            /** @} */
        };

        template<typename T_Type, uint32_t T_domainSize, uint32_t T_numWorkers, uint32_t T_simdSize>
        HDINLINE auto makeVar(ForEach<Config<T_domainSize, T_numWorkers, T_simdSize>> const& forEach)
        {
            return Variable<T_Type, typename ForEach<Config<T_domainSize, T_numWorkers, T_simdSize>>::BaseConfig>();
        }

#if 0
        template<
            typename T_Type,
            uint32_t T_domainSize,
            uint32_t T_numWorkers,
            uint32_t T_simdSize,
            typename... T_Args>
        HDINLINE auto makeVar(ForEach<Config<T_domainSize, T_numWorkers, T_simdSize>> const& forEach, T_Args&&... args)
        {
            return Variable<T_Type, typename ForEach<Config<T_domainSize, T_numWorkers, T_simdSize>>::BaseConfig>(
                std::forward<T_Args>(args)...);
        }
#endif
        template<typename T_Type, uint32_t T_domainSize, uint32_t T_numWorkers, uint32_t T_simdSize>
        HDINLINE auto makeVar(
            ForEach<Config<T_domainSize, T_numWorkers, T_simdSize>> const& forEach,
            T_Type const& args)
        {
            return Variable<T_Type, typename ForEach<Config<T_domainSize, T_numWorkers, T_simdSize>>::BaseConfig>(
                args);
        }

    } // namespace lockstep
} // namespace pmacc

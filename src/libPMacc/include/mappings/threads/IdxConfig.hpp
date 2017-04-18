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

namespace PMacc
{
namespace mappings
{
namespace threads
{

    /** describe a constant index domain
     *
     * @tparam T_domainSize number of elements in the domain
     * @tparam T_workerSize number of worker (threads) working on @p T_domainSize
     * @tparam T_simdSize 
     */
    template<
        uint32_t T_domainSize,
        uint32_t T_workerSize,
        uint32_t T_simdSize = 1
    >
    struct IdxConfig
    {
        static constexpr uint32_t domainSize = T_domainSize;
        static constexpr uint32_t workerSize = T_workerSize;
        static constexpr uint32_t simdSize = T_simdSize;

        static constexpr uint32_t collectiveOps =
            (( domainSize + simdSize * workerSize - 1 ) / ( simdSize * workerSize));
    };

} // namespace threads
} // namespace mappings
} // namespace PMacc

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

    template<
        uint32_t T_domainDim,
        uint32_t T_workerDim,
        uint32_t T_simdDim = 1
    >
    struct IdxConfig
    {
        static constexpr uint32_t domainDim = T_domainDim;
        static constexpr uint32_t workerDim = T_workerDim;
        static constexpr uint32_t simdDim = T_simdDim;

        static constexpr uint32_t collectiveOps =
            (( domainDim + simdDim * workerDim - 1 ) / ( simdDim * workerDim));
    };

} // namespace threads
} // namespace mappings
} // namespace PMacc

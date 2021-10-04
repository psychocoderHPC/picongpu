/* Copyright 2013-2022 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "pmacc/mappings/kernel/MappingDescription.hpp"

#include <cstdint>

namespace pmacc
{
    /** A shared memory on gpu. Used in conjunction with \ref pmacc::DataBox.
     *
     * @tparam T_TYPE type of memory objects
     * @tparam T_Vector CT::Vector with size description (per dimension)
     * @tparam T_id unique id for this object
     *              (is needed if more than one instance of shared memory in one kernel is used)
     * @tparam T_dim dimension of the memory (supports DIM1,DIM2 and DIM3)
     */
    template<
        typename T_TYPE,
        typename T_Vector,
        uint32_t T_id,
        typename T_MemoryMapping,
        uint32_t T_dim = T_Vector::dim>
    struct SharedBox
    {
        T_TYPE* fixedPointer;
    };
} // namespace pmacc

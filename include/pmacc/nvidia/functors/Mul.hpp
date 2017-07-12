/* Copyright 2014-2017 Axel Huebl
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
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"
#include "pmacc/mpi/GetMPI_Op.hpp"

namespace PMacc
{
namespace nvidia
{
namespace functors
{
    struct Mul
    {
        template<typename Dst, typename Src>
        HDINLINE void
        operator()( Dst& dst, const Src& src ) const
        {
            dst *= src;
        }
    };
} // namespace functors
} // namespace nvidia
} // namespace PMacc

namespace PMacc
{
namespace mpi
{
    template<>
    MPI_Op getMPI_Op<PMacc::nvidia::functors::Mul>()
    {
        return MPI_PROD;
    }
} // namespace mpi
} // namespace PMacc

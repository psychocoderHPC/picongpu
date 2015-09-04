/**
 * Copyright 2013 Heiko Burau
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

#include <math/vector/Int.hpp>  // math::Int
#include "types.h"

#include <utility>              // std::forward

namespace PMacc
{
namespace algorithm
{
namespace kernel
{
namespace detail
{
    class kernelForeach
    {
    public:
        //-----------------------------------------------------------------------------
        //! The kernel.
        //-----------------------------------------------------------------------------
        template<
            typename T_Acc,
            typename TMapper,
            typename TFunctor,
            typename... TC>
        ALPAKA_FN_ACC void operator()(
            T_Acc const & acc,
            TMapper const & mapper,
            TFunctor const & functor,
            TC && ... c) const
        {
            math::Int<TMapper::dim> cellIndex(
                mapper(
                    alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc),
                    alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)));

            functor(c[cellIndex]...);
        }
    };

} // namespace detail
} // namespace kernel
} // namespace algorithm
} // namespace PMacc

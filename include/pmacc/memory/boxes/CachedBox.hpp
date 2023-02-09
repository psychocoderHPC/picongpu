/* Copyright 2013-2022 Heiko Burau, Rene Widera
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

#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/SharedBox.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace CachedBox
    {
        template<uint32_t Id_, typename ValueType_, class BlockDescription_, typename T_Worker, typename Layout =  typename BlockDescription_::FullSuperCellSize>
        DINLINE constexpr auto create(
            T_Worker const& worker,
            const BlockDescription_ block,
            const Layout layout = Layout{})
        {
            constexpr uint32_t dim = BlockDescription_::FullSuperCellSize::dim;
            using OffsetOrigin = typename BlockDescription_::OffsetOrigin;
            using Type
                = DataBox<SharedBox<ValueType_, typename BlockDescription_::FullSuperCellSize, Id_, dim, Layout>>;
            return Type{Type::init(worker)}.shift(DataSpace<OffsetOrigin::dim>{OffsetOrigin::toRT()});
        }

        template<uint32_t Id_, typename ValueType_, class BlockDescription_, typename T_Worker, typename Layout =  typename BlockDescription_::FullSuperCellSize>
        DINLINE constexpr auto createProxy(
            T_Worker const& worker,
            const BlockDescription_ block,
            const Layout layout = Layout{})
        {
            constexpr uint32_t dim = BlockDescription_::FullSuperCellSize::dim;
            using OffsetOrigin = typename BlockDescription_::OffsetOrigin;
            using Type
                = DataBox<SharedBoxRead<ValueType_, typename BlockDescription_::FullSuperCellSize, Id_, dim, Layout>>;
            return Type{Type::init(worker)}.shift(DataSpace<OffsetOrigin::dim>{OffsetOrigin::toRT()});
        }

    } // namespace CachedBox
} // namespace pmacc

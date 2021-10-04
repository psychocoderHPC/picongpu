/* Copyright 2013-2022 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz
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

#include "SharedBox.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/shared/Allocate.hpp"

#include <llama/llama.hpp>

namespace pmacc
{
    template<typename Base>
    struct DataBox : Base
    {
        HDINLINE DataBox() = default;

        HDINLINE DataBox(Base base) : Base{std::move(base)}
        {
        }

        HDINLINE DataBox(DataBox const&) = default;

        HDINLINE decltype(auto) operator()(DataSpace<Base::Dim> const& idx = {}) const
        {
            if constexpr(Base::Dim == 1)
            {
                return (*this)[idx.x()];
            }
            else if constexpr(Base::Dim == 2)
            {
                return (*this)[idx.y()][idx.x()];
            }
            else if constexpr(Base::Dim == 3)
            {
                return (*this)[idx.z()][idx.y()][idx.x()];
            }
            else
            {
                static_assert(sizeof(idx) == 0, "Dim must be 1, 2 or 3");
            }
        }

        HDINLINE DataBox shift(DataSpace<Base::Dim> const& offset) const
        {
            DataBox result(*this);
            result.fixedPointer = &((*this)(offset));
            return result;
        }
    };

    namespace internal
    {
        template<typename... Sizes>
        HDINLINE auto toArrayExtents(math::CT::Vector<Sizes...>)
        {
            using V = math::CT::Vector<Sizes...>;
            using IndexType = typename math::CT::Vector<Sizes...>::type;
            if constexpr(V::dim == 1)
            {
                return llama::ArrayExtents<IndexType, V::x::value>{};
            }
            else if constexpr(V::dim == 2)
            {
                return llama::ArrayExtents<IndexType, V::y::value, V::x::value>{};
            }
            else if constexpr(V::dim == 3)
            {
                return llama::ArrayExtents<IndexType, V::z::value, V::y::value, V::x::value>{};
            }
            else
            {
                static_assert(sizeof(IndexType) == 0, "Vector dimension must be 1, 2 or 3");
            }
        }

        // LLAMA and DataSpace indices have the same semantic, fast moving index is first.
        template<unsigned Dim>
        HDINLINE auto toArrayIndex(DataSpace<Dim> idx)
        {
            using IndexType = typename DataSpace<Dim>::type;
            using ArrayIndex = llama::ArrayIndex<IndexType, Dim>;
            if constexpr(Dim == 1)
            {
                return ArrayIndex{idx[0]};
            }
            else if constexpr(Dim == 2)
            {
                return ArrayIndex{idx[1], idx[0]};
            }
            else if constexpr(Dim == 3)
            {
                return ArrayIndex{idx[2], idx[1], idx[0]};
            }
            else
            {
                static_assert(sizeof(idx) == 0, "Dim must be 1, 2 or 3");
            }
        }
    } // namespace internal

    // handle DataBox wrapping SharedBox with LLAMA
    template<typename T_TYPE, class T_SizeVector, typename T_MemoryMapping, uint32_t T_id, uint32_t T_dim>
    struct DataBox<SharedBox<T_TYPE, T_SizeVector, T_id, T_MemoryMapping, T_dim>>
    {
        using SB = SharedBox<T_TYPE, T_SizeVector, T_id, T_MemoryMapping, T_dim>;

        inline static constexpr std::uint32_t Dim = T_dim;
        using ValueType = T_TYPE;
        using Size = T_SizeVector;

        using SplitRecordDim = llama::TransformLeaves<T_TYPE, math::ReplaceVectorByArray>;
        using RecordDim = std::conditional_t<T_MemoryMapping::splitVector, SplitRecordDim, T_TYPE>;
        using ArrayExtents = decltype(internal::toArrayExtents(T_SizeVector{}));
        using Mapping = typename T_MemoryMapping::template fn<ArrayExtents, RecordDim>;
        using View = llama::View<Mapping, std::byte*>;

        View view;

        HDINLINE DataBox() = default;

        HDINLINE DataBox(SB sb)
            : view{
                Mapping{{}},
                llama::Array<std::byte*, 1>{
                    const_cast<std::byte*>(reinterpret_cast<const std::byte*>(sb.fixedPointer))}}
        {
        }

        HDINLINE decltype(auto) operator()(DataSpace<T_dim> idx = {}) const
        {
            auto&& ref = const_cast<View&>(view)(internal::toArrayIndex(DataSpace<T_dim>{idx + offset}));
            if constexpr(math::isVector<T_TYPE> && llama::isRecordRef<std::remove_reference_t<decltype(ref)>>)
                return math::makeVectorWithLlamaStorage<T_TYPE>(ref);
            else
                return ref;
        }

        HDINLINE DataBox shift(const DataSpace<T_dim>& offset) const
        {
            DataBox result(*this);
            result.offset += offset;
            return result;
        }

        template<typename T_Worker>
        static DINLINE SB init(T_Worker const& worker)
        {
            auto& mem_sh
                = memory::shared::allocate<T_id, memory::Array<ValueType, math::CT::volume<Size>::type::value>>(
                    worker);
            return {mem_sh.data()};
        }

        DataSpace<T_dim> offset{};
    };
} // namespace pmacc
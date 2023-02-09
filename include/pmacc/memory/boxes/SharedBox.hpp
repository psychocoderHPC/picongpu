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

#include "pmacc/math/Vector.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/memory/shared/Allocate.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace detail
    {
        template<typename T_Vector, typename T_TYPE>
        HDINLINE auto& subscript(T_TYPE* p, int const idx, std::integral_constant<uint32_t, 1>)
        {
            return p[idx];
        }

        template<typename T_Vector, typename T_TYPE>
        HDINLINE auto* subscript(T_TYPE* p, int const idx, std::integral_constant<uint32_t, 2>)
        {
            return p + idx * T_Vector::x::value;
        }

        template<typename T_Vector, typename T_TYPE>
        HDINLINE auto* subscript(T_TYPE* p, int const idx, std::integral_constant<uint32_t, 3>)
        {
            return p + idx * (T_Vector::x::value * T_Vector::y::value);
        }

        template<typename T_Vector, typename T_Result, typename T_TYPE>
        HDINLINE auto subscript2(T_TYPE* p, int const idx, std::integral_constant<uint32_t, 1>)
        {
            constexpr uint32_t dim = T_Result::dim;
            constexpr uint32_t volume = pmacc::math::CT::volume<T_Vector>::type::value;
            T_Result tmp;
            for(int i = 0; i < dim; ++i)
                *(reinterpret_cast<T_TYPE**>(&tmp) + i) = p + (idx + i * volume);
            return tmp;
        }

        template<typename T_Vector, typename T_Result, typename T_TYPE>
        HDINLINE auto* subscript2(T_TYPE* p, int const idx, std::integral_constant<uint32_t, 2>)
        {
            return p + idx * T_Vector::x::value;
        }

        template<typename T_Vector, typename T_Result, typename T_TYPE>
        HDINLINE auto* subscript2(T_TYPE* p, int const idx, std::integral_constant<uint32_t, 3>)
        {
            return p + idx * (T_Vector::x::value * T_Vector::y::value);
        }
    } // namespace detail

    /** create shared memory on gpu
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
        uint32_t T_id = 0,
        uint32_t T_dim = T_Vector::dim,
        typename T_LayoutNumElem = T_Vector>
    struct SharedBox
    {
        static constexpr bool proxy = false;
        static constexpr std::uint32_t Dim = T_dim;

        using ValueType = T_TYPE;
        using RefValueType = ValueType&;
        using Size = T_LayoutNumElem;

        HDINLINE
        SharedBox(ValueType* pointer = nullptr) : fixedPointer(pointer)
        {
        }

        HDINLINE SharedBox(SharedBox const&) = default;

        using ReducedType1D = T_TYPE&;
        using ReducedType2D = SharedBox<
            T_TYPE,
            math::CT::Int<T_Vector::x::value>,
            T_id,
            DIM1,
            math::CT::Int<T_LayoutNumElem::x::value>>;
        using ReducedType3D = SharedBox<
            T_TYPE,
            math::CT::Int<T_Vector::x::value, T_Vector::y::value>,
            T_id,
            DIM2,
            math::CT::Int<T_LayoutNumElem::x::value, T_LayoutNumElem::y::value>>;
        using ReducedType
            = std::conditional_t<Dim == 1, ReducedType1D, std::conditional_t<Dim == 2, ReducedType2D, ReducedType3D>>;

        HDINLINE ReducedType operator[](const int idx) const
        {
            ///@todo(bgruber): inline and replace this by if constexpr in C++17
            return {detail::subscript<T_LayoutNumElem>(fixedPointer, idx, std::integral_constant<uint32_t, T_dim>{})};
        }

        /*!return the first value in the box (list)
         * @return first value
         */
        HDINLINE RefValueType operator*()
        {
            return *fixedPointer;
        }

        HDINLINE ValueType const* getPointer() const
        {
            return fixedPointer;
        }
        HDINLINE ValueType* getPointer()
        {
            return fixedPointer;
        }

        /** create a shared memory box
         *
         * This call synchronizes a block and must be called from all threads and
         * not inside a if clauses
         */
        template<typename T_Worker>
        static DINLINE SharedBox init(T_Worker const& worker)
        {
            auto& mem_sh
                = memory::shared::allocate<T_id, memory::Array<ValueType, math::CT::volume<Size>::type::value>>(
                    worker);
            return {mem_sh.data()};
        }

    protected:
        PMACC_ALIGN(fixedPointer, ValueType*);
    };

    template<
        typename T_TYPE,
        typename T_Vector,
        uint32_t T_id = 0,
        uint32_t T_dim = T_Vector::dim,
        typename T_LayoutNumElem = T_Vector>
    struct SharedBoxRead;

    template<typename T, int T_valueDim, typename T_Vector, uint32_t T_id, uint32_t T_dim, typename T_LayoutNumElem>
    struct SharedBoxRead<math::Vector<T, T_valueDim>, T_Vector, T_id, T_dim, T_LayoutNumElem>
    {
        static constexpr bool proxy = true;
        static constexpr std::uint32_t Dim = T_dim;
        using T_TYPE = math::Vector<T, T_valueDim>;
        using ValueTypeProxy = math::Vector<
            T,
            T_valueDim,
            math::StandardAccessor,
            math::StandardNavigator,
            math::detail::Vector_proxy<T, T_valueDim>>;
        using ValueType = T_TYPE;
        using RefValueType = ValueTypeProxy;
        using Size = T_LayoutNumElem;

        HDINLINE
        SharedBoxRead(ValueType* pointer = nullptr) : fixedPointer(pointer)
        {
        }

        HDINLINE
        SharedBoxRead(T* pointer = nullptr) : fixedPointer(reinterpret_cast<ValueType*>(pointer))
        {
        }

        HDINLINE SharedBoxRead(SharedBoxRead const&) = default;

        using ReducedType1D = ValueTypeProxy;
        using ReducedType2D = SharedBoxRead<T_TYPE, math::CT::Int<T_Vector::x::value>, T_id, DIM1, T_LayoutNumElem>;
        using ReducedType3D = SharedBoxRead<
            T_TYPE,
            math::CT::Int<T_Vector::x::value, T_Vector::y::value>,
            T_id,
            DIM2,
            T_LayoutNumElem>;
        using ReducedType
            = std::conditional_t<Dim == 1, ReducedType1D, std::conditional_t<Dim == 2, ReducedType2D, ReducedType3D>>;

        HDINLINE ReducedType operator[](const int idx) const
        {
            ///@todo(bgruber): inline and replace this by if constexpr in C++17
            return {detail::subscript2<T_LayoutNumElem, ValueTypeProxy>(
                reinterpret_cast<T*>(fixedPointer),
                idx,
                std::integral_constant<uint32_t, T_dim>{})};
        }
#if 0
        /*!return the first value in the box (list)
         * @return first value
         */
        HDINLINE RefValueType operator*()
        {
            return *fixedPointer;
        }

        HDINLINE ValueType const* getPointer() const
        {
            return fixedPointer;
        }
        HDINLINE ValueType* getPointer()
        {
            return fixedPointer;
        }
#endif
        /** create a shared memory box
         *
         * This call synchronizes a block and must be called from all threads and
         * not inside a if clauses
         */
        template<typename T_Worker>
        static DINLINE SharedBoxRead init(T_Worker const& worker)
        {
            auto& mem_sh
                = memory::shared::allocate<T_id, memory::Array<ValueType, math::CT::volume<Size>::type::value>>(
                    worker);
            return {mem_sh.data()};
        }

    protected:
        PMACC_ALIGN(fixedPointer, ValueType*);
    };
} // namespace pmacc

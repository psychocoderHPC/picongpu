/**
 * Copyright 2013-2014 Heiko Burau, Rene Widera
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

#include "Vector.hpp"

namespace PMacc
{
namespace math
{

template<int dim>
struct Size_t : public Vector<size_t, dim>
{
    typedef Vector<size_t, dim> BaseType;

    HDINLINE Size_t()
    {
    }

    HDINLINE Size_t(size_t x) : BaseType(x)
    {
    }

    HDINLINE Size_t(size_t x, size_t y) : BaseType(x, y)
    {
    }

    HDINLINE Size_t(size_t x, size_t y, size_t z) : BaseType(x, y, z)
    {
    }

    /*! only allow explicit cast*/
    template<
    typename T_OtherType,
    typename T_OtherAccessor,
    typename T_OtherNavigator,
    template <typename, int> class T_OtherStorage>
    HDINLINE explicit Size_t(const Vector<T_OtherType, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& vec) :
    BaseType(vec)
    {
    }

    HDINLINE Size_t(const BaseType& vec) :
    BaseType(vec)
    {
    }
};

} // math
} // PMacc


namespace alpaka
{
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The DataSpace dimension get trait specialization.
            //#############################################################################
            template<
                int dim>
            struct DimType<
                PMacc::math::Size_t<dim>>
            {
                using type = alpaka::dim::DimInt<dim>;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The DataSpace extent get trait specialization.
            //#############################################################################
            template<
                typename T_Idx,
                int dim>
            struct GetExtent<
                T_Idx,
                PMacc::math::Size_t<dim>,
                typename std::enable_if<(dim > T_Idx::value)>::type>
            {
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    PMacc::math::Size_t<dim> const & extents)
                -> size::Size<PMacc::math::Size_t<dim>>
                {
                    return extents[(dim - 1u) - T_Idx::value];
                }
            };
            //#############################################################################
            //! The DataSpace extent set trait specialization.
            //#############################################################################
            template<
                typename T_Idx,
                int dim,
                typename T_Extent>
            struct SetExtent<
                T_Idx,
                PMacc::math::Size_t<dim>,
                T_Extent,
                typename std::enable_if<(dim > T_Idx::value)>::type>
            {
                ALPAKA_FN_HOST_ACC static auto setExtent(
                    PMacc::math::Size_t<dim> & extents,
                    T_Extent const & extent)
                -> void
                {
                    extents[(dim - 1u) - T_Idx::value] = extent;
                }
            };
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The Vector offset get trait specialization.
            //#############################################################################
            template<
                typename T_Idx,
                int dim>
            struct GetOffset<
                T_Idx,
                PMacc::math::Size_t<dim>,
                typename std::enable_if<(dim > T_Idx::value)>::type>
            {
                ALPAKA_FN_HOST_ACC static auto getOffset(
                    PMacc::math::Size_t<dim> const & offsets)
                -> size::Size<PMacc::math::Size_t<dim>>
                {
                    return offsets[(dim - 1u) - T_Idx::value];
                }
            };
            //#############################################################################
            //! The Vector offset set trait specialization.
            //#############################################################################
            template<
                typename T_Idx,
                int dim,
                typename T_Offset>
            struct SetOffset<
                T_Idx,
                PMacc::math::Size_t<dim>,
                T_Offset,
                typename std::enable_if<(dim > T_Idx::value)>::type>
            {
                ALPAKA_FN_HOST_ACC static auto setOffset(
                    PMacc::math::Size_t<dim> & offsets,
                    T_Offset const & offset)
                -> void
                {
                    offsets[(dim - 1u) - T_Idx::value] = offset;
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The Vector size type trait specialization.
            //#############################################################################
            template<
                int dim>
            struct SizeType<
                PMacc::math::Size_t<dim>>
            {
                using type = std::size_t;
            };
        }
    }
}

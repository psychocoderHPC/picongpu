/**
 * Copyright 2013-2015 Rene Widera, Benjamin Worpitz
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

#include "dimensions/DataSpace.hpp"

#include "traits/GetComponentsType.hpp"
#include "traits/GetNComponents.hpp"
#include "algorithms/math.hpp"
#include "algorithms/TypeCast.hpp"
#include "types.h"

namespace PMacc
{

namespace traits
{

template<unsigned DIM>
struct GetComponentsType<DataSpace<DIM>, false >
{
    typedef typename DataSpace<DIM>::type type;
};

/** Trait for float_X */
template<unsigned DIM>
struct GetNComponents<DataSpace<DIM>,false >
{
    static const uint32_t value=DIM;
};

}// namespace traits

namespace algorithms
{
namespace precisionCast
{

template<unsigned T_Dim>
struct TypeCast<int, PMacc::DataSpace<T_Dim> >
{
    typedef const PMacc::DataSpace<T_Dim>& result;

    HDINLINE result operator( )(const PMacc::DataSpace<T_Dim>& vector ) const
    {
        return vector;
    }
};

template<typename T_CastToType, unsigned T_Dim>
struct TypeCast<T_CastToType, PMacc::DataSpace<T_Dim>  >
{
    typedef ::PMacc::math::Vector<T_CastToType, T_Dim> result;

    HDINLINE result operator( )(const PMacc::DataSpace<T_Dim>& vector ) const
    {
        return result( vector );
    }
};

} //namespace typecast
} //namespace algorithms

} //namespace PMacc

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
                unsigned DIM>
            struct DimType<
                PMacc::DataSpace<DIM>>
            {
                using type = alpaka::dim::DimInt<DIM>;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The DataSpace width get trait specialization.
            //#############################################################################
            template<
                typename T_Idx,
                unsigned DIM>
            struct GetExtent<
                T_Idx,
                PMacc::DataSpace<DIM>,
                typename std::enable_if<(DIM > T_Idx::value)>::type>
            {
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    PMacc::DataSpace<DIM> const & extents)
                -> int
                {
                    return extents[(DIM - 1u) - T_Idx::value];
                }
            };
            //#############################################################################
            //! The DataSpace width set trait specialization.
            //#############################################################################
            template<
                typename T_Idx,
                unsigned DIM,
                typename T_Extent>
            struct SetExtent<
                T_Idx,
                PMacc::DataSpace<DIM>,
                T_Extent,
                typename std::enable_if<(DIM > T_Idx::value)>::type>
            {
                ALPAKA_FN_HOST_ACC static auto setExtent(
                    PMacc::DataSpace<DIM> & extents,
                    T_Extent const & extent)
                -> void
                {
                    extents[(DIM - 1u) - T_Idx::value] = extent;
                }
            };
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The Vec<DIM> offset get trait specialization.
            //#############################################################################
            template<
                typename T_Idx,
                unsigned DIM>
            struct GetOffset<
                T_Idx,
                PMacc::DataSpace<DIM>,
                typename std::enable_if<(DIM > T_Idx::value)>::type>
            {
                ALPAKA_FN_HOST_ACC static auto getOffset(
                    PMacc::DataSpace<DIM> const & offsets)
                -> int
                {
                    return offsets[(DIM - 1u) - T_Idx::value];
                }
            };
            //#############################################################################
            //! The Vec<DIM> offset set trait specialization.
            //#############################################################################
            template<
                typename T_Idx,
                unsigned DIM,
                typename T_Offset>
            struct SetOffset<
                T_Idx,
                PMacc::DataSpace<DIM>,
                T_Offset,
                typename std::enable_if<(DIM > T_Idx::value)>::type>
            {
                ALPAKA_FN_HOST_ACC static auto setOffset(
                    PMacc::DataSpace<DIM> & offsets,
                    T_Offset const & offset)
                -> void
                {
                    offsets[(DIM - 1u) - T_Idx::value] = offset;
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
                unsigned DIM>
            struct SizeType<
                PMacc::DataSpace<DIM>>
            {
                using type = int;
            };
        }
    }
}

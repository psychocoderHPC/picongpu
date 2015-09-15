/**
 * Copyright 2013-2015 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "math/vector/Size_t.hpp"
#include "types.h"

#include <boost/mpl/void.hpp>

namespace PMacc
{
namespace algorithm
{
namespace kernel
{
namespace detail
{

namespace mpl = boost::mpl;

/** The SphericMapper maps from cuda blockIdx and/or threadIdx to the cell index
 * \tparam dim dimension
 * \tparam BlockSize compile-time vector of the cuda block size (optional)
 * \tparam dummy neccesary to implement the optional BlockSize parameter
 *
 * If BlockSize is given the cuda variable blockDim is not used which is faster.
 */
template<int dim, typename BlockSize = mpl::void_, typename dummy = mpl::void_>
struct SphericMapper;

/* Compile-time BlockSize */

template<typename BlockSize>
struct SphericMapper<1, BlockSize>
{
    static const int dim = 1;

    DataSpace<dim> gridDim(const math::Size_t<dim>& size) const
    {
        return DataSpace<1>(size.x() / BlockSize::x::value);
    }

    HDINLINE
    math::Int<dim> operator()(const math::Int<dim>& _blockIdx,
                              const math::Int<dim>& _threadIdx) const
    {
        return _blockIdx.x() * BlockSize::x::value + _threadIdx.x();
    }

    HDINLINE
    math::Int<dim> operator()(const alpaka::Vec<alpaka::dim::DimInt<1u>, std::size_t>& _blockIdx, const alpaka::Vec<alpaka::dim::DimInt<1u>, std::size_t>& _threadIdx = alpaka::Vec<alpaka::dim::DimInt<1u>, std::size_t>(0)) const
    {
        return operator()(math::Int<dim>(_blockIdx[0]),
                          math::Int<dim>(_threadIdx[0]));
    }
};

template<typename BlockSize>
struct SphericMapper<2, BlockSize>
{
    static const int dim = 2;

    DataSpace<dim> gridDim(const math::Size_t<dim>& size) const
    {
        return DataSpace<dim>(size.x() / BlockSize::x::value,
                              size.y() / BlockSize::y::value);
    }

    HDINLINE
    math::Int<dim> operator()(const math::Int<dim>& _blockIdx,
                              const math::Int<dim>& _threadIdx) const
    {
        return math::Int<dim>( _blockIdx.x() * BlockSize::x::value + _threadIdx.x(),
                               _blockIdx.y() * BlockSize::y::value + _threadIdx.y() );
    }

    template<
        typename TSize>
    HDINLINE
    math::Int<dim> operator()(const alpaka::Vec<alpaka::dim::DimInt<2u>, std::size_t>& _blockIdx, const alpaka::Vec<alpaka::dim::DimInt<2u>, std::size_t>& _threadIdx = alpaka::Vec<alpaka::dim::DimInt<2u>, std::size_t>(0,0)) const
    {
        return operator()(math::Int<dim>(_blockIdx[0], _blockIdx[1]),
                          math::Int<dim>(_threadIdx[0], _threadIdx[1]));
    }
};

template<typename BlockSize>
struct SphericMapper<3, BlockSize>
{
    static const int dim = 3;

    DataSpace<dim> gridDim(const math::Size_t<dim>& size) const
    {
        return DataSpace<dim>(size.x() / BlockSize::x::value,
                            size.y() / BlockSize::y::value,
                            size.z() / BlockSize::z::value);
    }

    HDINLINE
    math::Int<dim> operator()(const math::Int<dim>& _blockIdx,
                             const math::Int<dim>& _threadIdx) const
    {
        return math::Int<dim>( _blockIdx * (math::Int<dim>)BlockSize().toRT() + _threadIdx );
    }

    template<
        typename TSize>
    HDINLINE
    math::Int<dim> operator()(const alpaka::Vec<alpaka::dim::DimInt<3u>, std::size_t>& _blockIdx, const alpaka::Vec<alpaka::dim::DimInt<3u>, std::size_t>& _threadIdx = alpaka::Vec<alpaka::dim::DimInt<3u>, std::size_t>(0,0,0)) const
    {
        return operator()(math::Int<dim>(_blockIdx[0], _blockIdx[1], _blockIdx[2]),
                          math::Int<dim>(_threadIdx[0], _threadIdx[1], _threadIdx[2]));
    }
};

} // detail
} // kernel
} // algorithm
} // PMacc

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
#include "cuSTL/cursor/BufferCursor.hpp"
#include "tag.h"

#include <stdint.h>
#include <memory>

namespace PMacc
{
namespace allocator
{

template<typename Type, int T_dim>
struct HostMemAllocator
{
    using MemBuf = alpaka::mem::buf::Buf<
        alpaka::dev::DevCpu,
        Type,
        alpaka::dim::DimInt<T_dim>,
        AlpakaSize>;

    typedef Type type;
    static const int dim = T_dim;
    typedef cursor::BufferCursor<type, T_dim> Cursor;
    typedef allocator::tag::host tag;

    HDINLINE
    cursor::BufferCursor<type, T_dim> allocate(const math::Size_t<T_dim>& size);
    template<typename TCursor>
    HDINLINE
    void deallocate(const TCursor& cursor);

    std::unique_ptr<MemBuf> m_upBuf;
};

} // allocator
} // PMacc

#include "HostMemAllocator.tpp"

/**
 * Copyright 2013 Heiko Burau, Rene Widera
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

#include "DeviceMemAllocator.hpp"

#include "Environment.hpp"

namespace PMacc
{
namespace allocator
{

template<typename Type, int T_dim>
cursor::BufferCursor<Type, T_dim>
DeviceMemAllocator<Type, T_dim>::allocate(const math::Size_t<T_dim>& size)
{
    assert(!m_upBuf);

    math::Size_t<T_dim-1> pitch;

#ifndef __CUDA_ARCH__
    m_upBuf.reset(new MemBuf(
        alpaka::mem::buf::alloc<Type, std::size_t>(
            Environment<>::get().DeviceManager().getDevice(),
            size)));

    if(dim == 2u)
    {
        pitch[0] = alpaka::mem::view::getPitchBytes<1u>(*m_upBuf.get());
    }
    else if(dim == 3u)
    {
        pitch[0] = alpaka::mem::view::getPitchBytes<2u>(*m_upBuf.get());
        pitch[1] = alpaka::mem::view::getPitchBytes<1u>(*m_upBuf.get());
    }
#endif

    return cursor::BufferCursor<Type, T_dim>(alpaka::mem::view::getPtrNative(*m_upBuf.get()), pitch);
}

template<typename Type, int T_dim>
template<typename TCursor>
void DeviceMemAllocator<Type, T_dim>::deallocate(const TCursor& cursor)
{
    // HACK: we ignore the cursor and hope it is the one created in the only allocate call.
    assert(cursor.getMarker() == alpaka::mem::view::getPtrNative(*m_upBuf.get()));
    m_upBuf.reset();
}

} // allocator
} // PMacc

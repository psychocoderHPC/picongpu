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

namespace PMacc
{
namespace allocator
{

template<typename Type, int T_dim>
cursor::BufferCursor<Type, T_dim>
DeviceMemEvenPitch<Type, T_dim>::allocate(const math::Size_t<T_dim>& size)
{
    math::Size_t<T_dim-1> pitch;

    // FIXME: Implement!

    return cursor::BufferCursor<Type, T_dim>(nullptr, pitch);
}

template<typename Type, int T_dim>
template<typename TCursor>
void DeviceMemEvenPitch<Type, T_dim>::deallocate(const TCursor& cursor)
{
    // FIXME: Implement!
}

} // allocator
} // PMacc

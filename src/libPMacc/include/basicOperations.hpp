/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, Rene Widera
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


#include <builtin_types.h>
#include "types.h"
#include <string>
#include <ostream>
#include <math_functions.h>
#include "nvidia/atomic.hpp"

namespace PMacc
{

template<typename T_Type>
DINLINE void atomicAddWrapper(T_Type* address, T_Type value)
{
    nvidia::detail::atomicAdd(address, value);
}

template<typename T_Type>
DINLINE void atomicAnyAdd(T_Type* address, T_Type value)
{

#if (__CUDA_ARCH__ >= 300)
    uint64_t key = reinterpret_cast<uint64_t>(address);
    uint32_t peers = nvidia::get_peers(key);
    nvidia::add_peers(address, value, peers);
#else
    nvidia::detail::atomicAdd(address, value);
#endif
}

} //namespace PMacc

/* CUDA STD structs and CPP STD ostream */
template <class T>
std::basic_ostream<T, std::char_traits<T> >& operator<<(std::basic_ostream<T, std::char_traits<T> >& out, const double3& v)
{
    out << "{" << v.x << " " << v.y << " " << v.z << "}";
    return out;
}

template <class T>
std::basic_ostream<T, std::char_traits<T> >& operator<<(std::basic_ostream<T, std::char_traits<T> >& out, const float3& v)
{
    out << "{" << v.x << " " << v.y << " " << v.z << "}";
    return out;
}



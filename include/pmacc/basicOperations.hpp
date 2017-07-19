/* Copyright 2013-2017 Felix Schmitt, Heiko Burau, Rene Widera
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


#include "pmacc/types.hpp"
#include <string>
#include <ostream>

namespace pmacc
{
namespace
{

DINLINE void atomicAddWrapper(float* address, float value)
{
    atomicAdd(address, value);
}

DINLINE void atomicAddWrapper(double* inAddress, double value)
{
#if( PMACC_CUDA_ENABLED == 1 )
    uint64_cu* address = (uint64_cu*) inAddress;
    double old = value;
    while (
           (old = __longlong_as_double(atomicExch(address,
                                                  (uint64_cu) __double_as_longlong(__longlong_as_double(atomicExch(address, (uint64_cu) 0L)) +
                                                                                   old)))) != 0.0);
#else
    atomicAdd(address, value);
#endif
}

}

} //namespace pmacc

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



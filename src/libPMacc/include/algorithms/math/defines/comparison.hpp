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

namespace PMacc
{
namespace algorithms
{

namespace math
{

template<typename T1,typename T2>
struct Max;

template<typename T1,typename T2>
struct Min;


template<typename T1,typename T2>
HDINLINE typename Min< T1,T2>::result min(const T1& value1,const T2& value2)
{
    return Min< T1,T2 > ()(value1,value2);
}

template<typename T1,typename T2>
HDINLINE typename Max< T1,T2 >::result max(const T1& value1,const T2& value2)
{
    return Max< T1,T2 > ()(value1,value2);
}

#ifndef __CUDA_ARCH__
    #include <algorithm>
#endif

template<>
struct Min<int, int>
{
    typedef int result;

    HDINLINE int operator()(int value1, int value2)
    {
#ifdef __CUDA_ARCH__ /*device version*/
        return ::min(value1, value2);
#else
        return std::min(value1, value2);
#endif
    }
};

template<>
struct Min<unsigned int, unsigned int>
{
    typedef unsigned int result;

    HDINLINE unsigned int operator()(unsigned int value1, unsigned int value2)
    {
#ifdef __CUDA_ARCH__ /*device version*/
        return ::umin(value1, value2);
#else
        return std::min(value1, value2);
#endif
    }
};

template<>
struct Min<long long int, long long int>
{
    typedef long long int result;

    HDINLINE long long int operator()(long long int value1, long long int value2)
    {
#ifdef __CUDA_ARCH__ /*device version*/
        return ::llmin(value1, value2);
#else
        return std::min(value1, value2);
#endif
    }
};

template<>
struct Min<unsigned long long int, unsigned long long int>
{
    typedef unsigned long long int result;

    HDINLINE unsigned long long int operator()(unsigned long long int value1, unsigned long long int value2)
    {
#ifdef __CUDA_ARCH__ /*device version*/
        return ::ullmin(value1, value2);
#else
        return std::min(value1, value2);
#endif
    }
};

template<>
struct Max<int, int>
{
    typedef int result;

    HDINLINE int operator()(int value1, int value2)
    {
#ifdef __CUDA_ARCH__ /*device version*/
        return ::max(value1, value2);
#else
        return std::max(value1, value2);
#endif
    }
};

template<>
struct Max<unsigned int, unsigned int>
{
    typedef unsigned int result;

    HDINLINE unsigned int operator()(unsigned int value1, unsigned int value2)
    {
#ifdef __CUDA_ARCH__ /*device version*/
        return ::umax(value1, value2);
#else
        return std::max(value1, value2);
#endif
    }
};

template<>
struct Max<long long int, long long int>
{
    typedef long long int result;

    HDINLINE long long int operator()(long long int value1, long long int value2)
    {
#ifdef __CUDA_ARCH__ /*device version*/
        return ::llmax(value1, value2);
#else
        return std::max(value1, value2);
#endif
    }
};

template<>
struct Max<unsigned long long int, unsigned long long int>
{
    typedef unsigned long long int result;

    HDINLINE unsigned long long int operator()(unsigned long long int value1, unsigned long long int value2)
    {
#ifdef __CUDA_ARCH__ /*device version*/
        return ::ullmax(value1, value2);
#else
        return std::max(value1, value2);
#endif
    }
};

} //namespace math
} //namespace algorithms
}//namespace PMacc

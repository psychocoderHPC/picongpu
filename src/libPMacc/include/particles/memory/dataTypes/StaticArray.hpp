/**
 * Copyright 2013 Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
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

#include "types.h"
#include "math/Vector.hpp"


namespace PMacc
{

namespace pmath = PMacc::math;
namespace pmacc = PMacc;

template<typename T_Type, typename T_size>
class StaticArray
{
public:
    static const uint32_t size = T_size::value;
    typedef T_Type Type;
private:
    Type data[size];
public:

    template<class> struct result;

    template<class F, typename TKey>
    struct result<F(TKey)>
    {
        typedef Type& type;
    };

    template<class F, typename TKey>
    struct result<const F(TKey)>
    {
        typedef const Type& type;
    };

    HDINLINE
    Type& operator[](const int idx)
    {
        return data[idx];
    }

    HDINLINE
    const Type& operator[](const int idx) const
    {
        return data[idx];
    }
};



template<typename T_Type, typename T_size, int T_dim,typename T_Accessor,typename T_Navigator,template<typename,int> class T_Storage>
class StaticArray<PMacc::math::Vector< T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>,T_size>
{
public:
    static const uint32_t size = T_size::value;
    typedef T_Type Type;


    typedef StaticHandle<Type,T_size,T_dim> Handle;
    Type data[size*T_dim];

public:

    template<class> struct result;

    template<class F, typename TKey>
    struct result<F(TKey)>
    {
        typedef Handle& type;
    };

    template<class F, typename TKey>
    struct result<const F(TKey)>
    {
        typedef const Handle& type;
    };

    HDINLINE
    Handle& operator[](const int idx)
    {
        return *((Handle*)(data+idx));//Handle(data+idx*T_dim);
    }

    HDINLINE
    const Handle& operator[](const int idx) const
    {
        return *((Handle*)(data+idx)); //Handle(data+idx*T_dim);
    }
};

} //namespace PMacc

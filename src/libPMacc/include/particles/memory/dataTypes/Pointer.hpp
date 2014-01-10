/**
 * Copyright 2014  Rene Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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


namespace PMacc
{

template <class TYPE>
class Pointer
{
public:

    typedef TYPE type;

    HDINLINE Pointer() : ptr(NULL)
    {
    }

    HDINLINE Pointer(type * const ptrIn) : ptr(ptrIn)
    {
    }

    HDINLINE Pointer(const Pointer<type>& other) : ptr(other.ptr)
    {
    }

    HDINLINE type& operator*()
    {
        return *ptr;
    }

    HDINLINE type* operator->()
    {
        return ptr;
    }

    HDINLINE bool operator==(const Pointer<type>& other) const
    {
        return ptr == other.ptr;
    }

    HDINLINE bool operator!=(const Pointer<type>& other) const
    {
        return ptr != other.ptr;
    }

    HDINLINE bool isValid() const
    {
        return ptr != NULL;
    }

    type * ptr;
};

} //namespace PMacc

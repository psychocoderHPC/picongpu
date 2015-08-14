/**
 * Copyright 2015  Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
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
#include "particles/memory/dataTypes/Pointer.hpp"


namespace PMacc
{

template<typename T_Type>
struct GetNoInitType;

/** wrapper for native C pointer
 *
 * @tparam T_Type type of the pointed object
 */
template <typename T_Type, typename T_InitMethod = detail::InitWithNULL>
class FramePointer : public Pointer<T_Type, T_InitMethod>
{
private:
    typedef Pointer<T_Type, T_InitMethod> Base;
public:
    typedef typename Base::type type;
    typedef typename Base::PtrType PtrType;

    /** default constructor
     *
     * the default pointer points to invalid memory
     */
    HDINLINE FramePointer( ) : Base( )
    {
    }

    HDINLINE FramePointer( PtrType const ptrIn ) : Base( ptrIn )
    {
    }

    HDINLINE FramePointer( const Base& other ) : Base( other )
    {
    }

    HDINLINE FramePointer( const FramePointer<type>& other ) : Base( other )
    {
    }

    template<typename T_OtherInitMethod>
    HDINLINE FramePointer( const FramePointer<type, T_OtherInitMethod>& other ) : Base( other )
    {
    }

    template<typename T_OtherInitMethod>
    HDINLINE FramePointer& operator=(const FramePointer<type, T_OtherInitMethod>& other)
    {
        Base::operator=(other);
        return *this;
    }

    HDINLINE typename type::ParticleType operator[](const uint32_t idx)
    {
        return (*Base::ptr)[idx];
    }

    HDINLINE const typename type::ParticleType operator[](const uint32_t idx) const
    {
        return (*Base::ptr)[idx];
    }

};

template<typename T_Type, typename T_InitMethod>
struct GetNoInitType<FramePointer<T_Type, T_InitMethod> >
{
    typedef FramePointer<T_Type, detail::NoInit> type;
};

} //namespace PMacc

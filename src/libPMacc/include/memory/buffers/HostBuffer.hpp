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


#ifndef _HOSTBUFFER_HPP
#define	_HOSTBUFFER_HPP

#include "memory/buffers/Buffer.hpp"
#include "dimensions/DataSpace.hpp"

namespace PMacc
{

class EventTask;

template <typename>
class DeviceBuffer;

/**
 * Interface for a DIM-dimensional Buffer of type TYPE on the host
 *
 * @tparam TYPE datatype for buffer data
 * @tparam DIM dimension of the buffer
 */
template <typename T_BufferDef>
class HostBuffer : public Buffer<T_BufferDef>
{
public:

    typedef T_BufferDef BufferDef;
    typedef typename BufferDef::ValueType TYPE;
    static const unsigned int DIM = BufferDef::dim;

    typedef HostBuffer<BufferDef> This;
    typedef Buffer<BufferDef> Base;

    /**
     * Copies the data from the given DeviceBuffer to this HostBuffer.
     *
     * @param other DeviceBuffer to copy data from
     */
    virtual void copyFrom(DeviceBuffer<BufferDef>& other) = 0;

    /**
     * Destructor.
     */
    virtual ~HostBuffer()
    {
    };

protected:

    /**
     * Constructor.
     *
     * @param dataSpace size of each dimension of the buffer
     */
    HostBuffer(DataSpace<DIM> dataSpace) :
    Base(dataSpace)
    {

    }
};

} //namespace PMacc


#endif	/* _HOSTBUFFER_HPP */

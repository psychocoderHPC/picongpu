/**
 * Copyright 2013 Heiko Burau, Rene Widera
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

#include <cassert>
#include <limits>

#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/PitchedBox.hpp"
#include "Environment.hpp"

#include "memory/dataTypes/ConstBufferSize.hpp"
#include "memory/dataTypes/BufferSize.hpp"

namespace PMacc
{

template <class T_Type, unsigned T_dim, class T_SizeDefinition = ConstBufferSize>
struct BufferDefinition
{
    typedef T_Type ValueType;
    static const unsigned int dim = T_dim;
    typedef T_SizeDefinition SizeDefinition;
};

/**
 * Minimal function description of a buffer,
 *
 * @tparam TYPE datatype stored in the buffer
 * @tparam DIM dimension of the buffer (1-3)
 */
template <typename T_BufferDef>
class Buffer : public T_BufferDef::SizeDefinition
{
public:

    typedef T_BufferDef BufferDef;
    typedef typename BufferDef::ValueType TYPE;
    static const unsigned int DIM = BufferDef::dim;
    typedef typename BufferDef::SizeDefinition SizeDefinition;

    typedef DataBox<PitchedBox<TYPE, DIM> > DataBoxType;

    /**
     * constructor
     * @param dataSpace description of spread of any dimension
     */
    Buffer(DataSpace<DIM> dataSpace) :
    SizeDefinition(dataSpace.productOfComponents(), dataSpace.productOfComponents()),
    data_space(dataSpace),
    data1D(true)
    {

    }

    /**
     * destructor
     */
    virtual ~Buffer()
    {

    }

    /*! Get base pointer to memory
     * @return pointer to this buffer in memory
     */
    virtual TYPE* getBasePointer() = 0;

    /*! Get pointer that includes all offsets
     * @return pointer to a point in a memory array
     */
    virtual TYPE* getPointer() = 0;

    /*! Get max spread (elements) of any dimension
     * @return spread (elements) per dimension
     */
    virtual DataSpace<DIM> getDataSpace() const
    {
        return data_space;
    }

    virtual DataSpace<DIM> getCurrentDataSpace()
    {
        return getCurrentDataSpace(this->getCurrentSize());
    }

    /*! Spread of memory per dimension which is currently used
     * @return if DIM == DIM1 than return count of elements (x-direction)
     * if DIM == DIM2 than return how many lines (y-direction) of memory is used
     * if DIM == DIM3 than return how many slides (z-direction) of memory is used
     */
    virtual DataSpace<DIM> getCurrentDataSpace(size_t currentSize)
    {
        DataSpace<DIM> tmp;
        int64_t current_size = static_cast<int64_t> (currentSize);

        //!\todo: current size can be changed if it is a DeviceBuffer and current size is on device
        //call first get current size (but const not allow this)

        if (DIM == DIM1)
        {
            tmp[0] = current_size;
        }
        if (DIM == DIM2)
        {
            if (current_size <= data_space[0])
            {
                tmp[0] = current_size;
                tmp[1] = 1;
            }
            else
            {
                tmp[0] = data_space[0];
                tmp[1] = (current_size + data_space[0] - 1) / data_space[0];
            }
        }
        if (DIM == DIM3)
        {
            if (current_size <= data_space[0])
            {
                tmp[0] = current_size;
                tmp[1] = 1;
                tmp[2] = 1;
            }
            else if (current_size <= (data_space[0] * data_space[1]))
            {
                tmp[0] = data_space[0];
                tmp[1] = (current_size + data_space[0] - 1) / data_space[0];
                tmp[2] = 1;
            }
            else
            {
                tmp[0] = data_space[0];
                tmp[1] = data_space[1];
                tmp[2] = (current_size + (data_space[0] * data_space[1]) - 1) / (data_space[0] * data_space[1]);
            }
        }

        return tmp;
    }

    virtual void reset(bool preserveData = false) = 0;

    virtual void setValue(const TYPE& value) = 0;

    virtual DataBox<PitchedBox<TYPE, DIM> > getDataBox() = 0;

    inline bool is1D()
    {
        return data1D;
    }

protected:

    /*! Check if my DataSpace is greater than other.
     * @param other other DataSpace
     * @return true if my DataSpace (one dimension) is greater than other, false otherwise
     */
    virtual bool isMyDataSpaceGreaterThan(DataSpace<DIM> other)
    {
        return !other.isOneDimensionGreaterThan(data_space);
    }

    DataSpace<DIM> data_space;

    bool data1D;

};

} //namespace PMacc

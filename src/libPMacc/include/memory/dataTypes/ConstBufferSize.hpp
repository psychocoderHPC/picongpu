/**
 * Copyright 2014 Rene Widera
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
#include <cassert>

namespace PMacc
{

class ConstBufferSize
{
public:

    ConstBufferSize(size_t sizeIn, size_t maxSizeIn) : size(sizeIn)
    {
        assert(sizeIn == maxSizeIn);
    }

    virtual ~ConstBufferSize()
    {
    }

    /*! returns the current size (count of elements)
     * @return current size
     */
    virtual size_t getCurrentSize()
    {
        return size;
    }

    /*! ignore any set call but checks that new size is equal than old size
     * @param newsize new current size
     */
    virtual void setCurrentSize(size_t newsize)
    {
        assert(newsize == size);
    }

private:

    const size_t size;
};

} //namespace PMacc

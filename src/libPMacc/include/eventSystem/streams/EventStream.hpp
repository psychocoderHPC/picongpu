/**
 * Copyright 2013-2015 Felix Schmitt, Rene Widera, Benjamin Worpitz
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

#include "eventSystem/events/CudaEvent.hpp"
#include "types.h"

#include <alpaka/alpaka.hpp>

namespace PMacc
{

/**
 * Wrapper for a single cuda stream.
 * Allows recording cuda events on the stream.
 */
class EventStream
{
public:

    /**
     * Constructor.
     * Creates the cudaStream_t object.
     */
    EventStream(alpaka::dev::Dev<AlpakaAccDev> dev) :
        stream(new AlpakaAccStream(dev))
    {}

    /**
     * Destructor.
     * Waits for the stream to finish and destroys it.
     */
    virtual ~EventStream()
    {
        //wait for all kernels in stream to finish
        alpaka::wait::wait(*stream);
    }

    /**
     * Returns the cudaStream_t object associated with this EventStream.
     * @return the internal cuda stream object
     */
    AlpakaAccStream& getCudaStream()
    {
        return *stream;
    }

    void waitOn(const CudaEvent& ev)
    {
        if(getCudaStream() != ev.getCudaStream())
        {
            alpaka::wait::wait(*stream, *ev);
        }
    }

private:
    AlpakaAccStream* stream;
};

}

/**
 * Copyright 2014 Rene Widera, Benjamin Worpitz
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

#include "types.h"
#include <alpaka/alpaka.hpp>

namespace PMacc
{

/**
 * Wrapper for cuda events
 */
class CudaEvent
{
public:
    /**
     * Constructor
     *
     * no data is allocated @see create()
     */
    CudaEvent() :
        m_event(NULL),
        m_pStream(NULL),
        isRecorded(false),
        isValid(false)
    {}


    /**
     * Destructor
     *
     * no data is freed @see destroy()
     */
    virtual ~CudaEvent()
    {

    }

    /**
     *  create valid object
     *
     * - internal memory is allocated
     * - event must be destroyed with @see destroy
     */
    static CudaEvent create(AlpakaAccDev const & dev)
    {
        CudaEvent ev;
        ev.m_event = new alpaka::event::Event<AlpakaAccStream>(dev);
        ev.isValid = true;
        return ev;
    }

    /**
     * free allocated memory
     */
    static void destroy(CudaEvent& ev)
    {
        alpaka::wait::wait(*ev.m_event);
        ev.isValid = false;
        delete ev.m_event;
        ev.m_event = NULL;
    }

    /**
     * get native cuda event
     *
     * @return native cuda event
     */
    alpaka::event::Event<AlpakaAccStream> operator*() const
    {
        assert(isValid);
        return *m_event;
    }

    /**
     * check whether the event is finished
     *
     * @return true if event is finished else false
     */
    bool isFinished() const
    {
        assert(isValid);
        return alpaka::event::test(*m_event);
    }

    /**
     * get stream in which this event is recorded
     *
     * @return native cuda stream
     */
    AlpakaAccStream& getCudaStream() const
    {
        assert(isRecorded);
        assert(m_pStream);
        return *m_pStream;
    }

    /**
     * record event in a device stream
     *
     * @param stream native cuda stream
     */
    void recordEvent( AlpakaAccStream* stream)
    {
        /* disallow double recording */
        assert(isRecorded==false);
        isRecorded = true;
        m_pStream = stream;
        alpaka::stream::enqueue(*m_pStream, *m_event);
    }

private:
    alpaka::event::Event<AlpakaAccStream>* m_event;
    AlpakaAccStream* m_pStream;
    /* state if event is recorded */
    bool isRecorded;
    bool isValid;
};
}

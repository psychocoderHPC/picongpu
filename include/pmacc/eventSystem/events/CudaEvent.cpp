/* Copyright 2016-2023 Rene Widera
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


#include "pmacc/eventSystem/events/CudaEvent.hpp"

#include "pmacc/Environment.hpp"
#include "pmacc/acc.hpp"
#include "pmacc/cuplaHelper/Device.hpp"
#include "pmacc/eventSystem/events/CudaEventHandle.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    CudaEvent::CudaEvent() : event(AlpakaEventType(manager::Device<AccDev>::get().current()))
    {
    }


    CudaEvent::~CudaEvent()
    {
        PMACC_ASSERT(refCounter == 0u);
        log(ggLog::CUDA_RT() + ggLog::EVENT(), "sync and delete event");
        // free cupla event
        alpaka::wait(event);
    }

    void CudaEvent::registerHandle()
    {
        ++refCounter;
    }

    void CudaEvent::releaseHandle()
    {
        assert(refCounter != 0u);
        // get old value and decrement
        uint32_t oldCounter = refCounter--;
        if(oldCounter == 1u)
        {
            // reset event meta data
            stream.reset();
            finished = true;

            Environment<>::get().EventPool().push(this);
        }
    }


    bool CudaEvent::isFinished()
    {
        // avoid alpaka calls if event is already finished
        if(!finished)
        {
            assert(stream);
            finished = alpaka::isComplete(event);
        }
        return finished;
    }


    void CudaEvent::recordEvent(AccStream const& stream)
    {
        /* disallow double recording */
        assert(!this->stream);
        finished = false;
        this->stream = stream;
        alpaka::enqueue(*this->stream, event);
    }

} // namespace pmacc

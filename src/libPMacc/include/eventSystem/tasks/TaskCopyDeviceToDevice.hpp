/**
 * Copyright 2013-2015 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz
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

#include "eventSystem/EventSystem.hpp"
#include "eventSystem/streams/EventStream.hpp"
#include "eventSystem/tasks/StreamTask.hpp"
#include "types.h"

#include <alpaka/alpaka.hpp>

namespace PMacc
{

    template <class TYPE, unsigned DIM>
    class DeviceBuffer;

    template <class TYPE, unsigned DIM>
    class TaskCopyDeviceToDevice : public StreamTask
    {
    public:

        TaskCopyDeviceToDevice( DeviceBuffer<TYPE, DIM>& src, DeviceBuffer<TYPE, DIM>& dst) :
        StreamTask()
        {
            this->source = & src;
            this->destination =  & dst;
        }

        virtual ~TaskCopyDeviceToDevice()
        {
            notify(this->myId, COPYDEVICE2DEVICE, NULL);
        }

        bool executeIntern()
        {
            return isFinished();
        }

        void event(id_t, EventType, IEventData*)
        {

        }

        virtual void init()
        {
            // __startAtomicTransaction( __getTransactionEvent());
            size_t current_size = source->getCurrentSize();
            destination->setCurrentSize(current_size);
            DataSpace<DIM> devCurrentSize = source->getCurrentDataSpace(current_size);

            alpaka::mem::view::copy(
                this->getEventStream()->getCudaStream(),
                this->destination->getMemBufView(),
                this->source->getMemBufView(),
                devCurrentSize);

            this->activate();
            // __setTransactionEvent(__endTransaction());
        }

        std::string toString()
        {
            return "TaskCopyDeviceToDevice";
        }

    protected:
        DeviceBuffer<TYPE, DIM> *source;
        DeviceBuffer<TYPE, DIM> *destination;
    };

} //namespace PMacc

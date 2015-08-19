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

#include <alpaka/alpaka.hpp>

namespace PMacc
{

    template <class TYPE, unsigned DIM>
    class HostBuffer;
    template <class TYPE, unsigned DIM>
    class DeviceBuffer;

    template <class TYPE, unsigned DIM>
    class TaskCopyHostToDevice : public StreamTask
    {
    public:

        TaskCopyHostToDevice(HostBuffer<TYPE, DIM>& src, DeviceBuffer<TYPE, DIM>& dst) :
        StreamTask()
        {
            this->host =  & src;
            this->device =  & dst;
        }

        virtual ~TaskCopyHostToDevice()
        {
            notify(this->myId, COPYHOST2DEVICE, NULL);
            //std::cout<<"destructor TaskH2D"<<std::endl;
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
            // __startAtomicTransaction(__getTransactionEvent());
            size_t current_size = host->getCurrentSize();
            DataSpace<DIM> hostCurrentSize = host->getCurrentDataSpace(current_size);
            alpaka::mem::view::copy(
                this->getEventStream()->getCudaStream(),
                this->device->getMemBufView(),
                this->host->getMemBufView(),
                hostCurrentSize);
            device->setCurrentSize(current_size);
            this->activate();
            // __setTransactionEvent(__endTransaction());
        }

        std::string toString()
        {
            return "TaskCopyHostToDevice";
        }

    protected:
        HostBuffer<TYPE, DIM> *host;
        DeviceBuffer<TYPE, DIM> *device;
    };


} //namespace PMacc

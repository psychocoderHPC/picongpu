/**
 * Copyright 2013-2016 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
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

#include "eventSystem/tasks/ITask.hpp"
#include "eventSystem/tasks/MPITask.hpp"
#include "eventSystem/tasks/TaskCopyHostToDevice.hpp"
#include "eventSystem/events/EventDataReceive.hpp"
#include "eventSystem/tasks/Factory.hpp"
#include "mappings/simulation/EnvironmentController.hpp"
#include "memory/buffers/Exchange.hpp"

namespace PMacc
{


    template <class TYPE, unsigned DIM>
    class TaskReceive : public MPITask
    {
    public:

        TaskReceive(Exchange<TYPE, DIM> &ex) :
        exchange(&ex),
        state(Constructor)
        {
        }

        virtual void init()
        {
            state = WaitForReceived;
            Environment<>::get().Factory().createTaskReceiveMPI(exchange, this);
        }

        bool executeIntern()
        {
            switch (state)
            {
                case WaitForReceived:
                    break;
                case RunCopy:
                    state = WaitForFinish;
                   __startTransaction();
                    if (exchange->hasDeviceDoubleBuffer())
                    {
#if( PMACC_ENABLE_GPUDIRECT == 0 )
                        exchange->getHostBuffer().setCurrentSize(newBufferSize);
                        Environment<>::get().Factory().createTaskCopyHostToDevice(exchange->getHostBuffer(),
                                                                                     exchange->getDeviceDoubleBuffer());
#else
                        // since we had no host buffer we need to set the buffer size
                        exchange->getDeviceDoubleBuffer().setCurrentSize(newBufferSize);
#endif
                        Environment<>::get().Factory().createTaskCopyDeviceToDevice(exchange->getDeviceDoubleBuffer(),
                                                                                       exchange->getDeviceBuffer(),
                                                                                       this);
                    }
                    else
                    {
#if( PMACC_ENABLE_GPUDIRECT == 0 )
                        exchange->getHostBuffer().setCurrentSize(newBufferSize);
                        Environment<>::get().Factory().createTaskCopyHostToDevice(exchange->getHostBuffer(),
                                                                                     exchange->getDeviceBuffer(),
                                                                                     this);
#else
                        // set destination buffer size
                        exchange->getDeviceBuffer().setCurrentSize(newBufferSize);
                        setSizeEvent=__getTransactionEvent();
                        state = WaitForSetSize;
#endif
                    }
                    __endTransaction();
                    break;
                case WaitForSetSize:
                    // this code is only passed if PMACC_ENABLE_GPUDIRECT != 0
                    if(NULL == Environment<>::get().Manager().getITaskIfNotFinished(setSizeEvent.getTaskId()))
                    {
                        state = Finish;
                        return true;
                    }
                    break;
                case WaitForFinish:
                    break;
                case Finish:
                    return true;
                default:
                    return false;
            }

            return false;
        }

        virtual ~TaskReceive()
        {
            notify(this->myId, RECVFINISHED, NULL);
        }

        void event(id_t, EventType type, IEventData* data)
        {
            switch (type)
            {
                case RECVFINISHED:
                    if (data != NULL)
                    {
                        EventDataReceive *rdata = static_cast<EventDataReceive*> (data);
                        // std::cout<<" data rec "<<rdata->getReceivedCount()/sizeof(TYPE)<<std::endl;
                        newBufferSize = rdata->getReceivedCount() / sizeof (TYPE);
                        state = RunCopy;
                        executeIntern();
                    }
                    break;
                case COPYHOST2DEVICE:
                case COPYDEVICE2DEVICE:
                    state = Finish;
                    break;
                default:
                    return;
            }
        }

        std::string toString()
        {
            std::stringstream ss;
            ss<<state;
            return std::string("TaskReceive ")+ ss.str();
        }

    private:

        enum state_t
        {
            Constructor,
            WaitForReceived,
            RunCopy,
            WaitForSetSize,
            WaitForFinish,
            Finish

        };


        Exchange<TYPE, DIM> *exchange;
        state_t state;
        size_t newBufferSize;
        EventTask setSizeEvent;
    };

} //namespace PMacc


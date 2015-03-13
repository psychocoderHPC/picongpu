/**
 * Copyright 2013 Felix Schmitt, Rene Widera, Wolfgang Hoenig
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

#ifndef _TASKRECEIVE_HPP
#define	_TASKRECEIVE_HPP

#include "memory/buffers/Exchange.hpp"

#include "mappings/simulation/EnvironmentController.hpp"
#include "eventSystem/tasks/ITask.hpp"
#include "eventSystem/tasks/MPITask.hpp"
#include "eventSystem/tasks/TaskCopyHostToDevice.hpp"
#include "eventSystem/events/EventDataReceive.hpp"
#include "eventSystem/tasks/Factory.hpp"

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
                    __startAtomicTransaction();
                    if (exchange->hasDeviceDoubleBuffer())
                    {

#if(PMACC_ENABLE_GPUDIRECT==0)
                        exchange->getHostBuffer().setCurrentSize(newBufferSize);
                        Environment<>::get().Factory().createTaskCopyHostToDevice(exchange->getHostBuffer(),
                                                                                     exchange->getDeviceDoubleBuffer());
#else
                        exchange->getDeviceDoubleBuffer().setCurrentSize(newBufferSize);
#endif
                        Environment<>::get().Factory().createTaskCopyDeviceToDevice(exchange->getDeviceDoubleBuffer(),
                                                                                       exchange->getDeviceBuffer(),
                                                                                       this);
                    }
                    else
                    {
#if(PMACC_ENABLE_GPUDIRECT==0)
                        exchange->getHostBuffer().setCurrentSize(newBufferSize);
                        Environment<>::get().Factory().createTaskCopyHostToDevice(exchange->getHostBuffer(),
                                                                                     exchange->getDeviceBuffer(),
                                                                                     this);
#else
                        exchange->getDeviceBuffer().setCurrentSize(newBufferSize);
                        setSizeEvent=__getTransactionEvent();
                        state = WaitForSetSize;
#endif

                    }
                    __endTransaction();
                    break;
                case WaitForFinish:
                    break;
#if(PMACC_ENABLE_GPUDIRECT==1)
                case WaitForSetSize:
                    if (NULL == Environment<>::get().Manager().getITaskIfNotFinished(setSizeEvent.getTaskId()))
                    {
                        state = Finish;
                        return true;
                    }
                    break;
#endif
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
                        __startTransaction(); //no blocking
                        EventDataReceive *rdata = static_cast<EventDataReceive*> (data);
                        // std::cout<<" data rec "<<rdata->getReceivedCount()/sizeof(TYPE)<<std::endl;
                        newBufferSize = rdata->getReceivedCount() / sizeof (TYPE);
                        __endTransaction();
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
            std::stringstream dbuff;
            dbuff<<exchange->hasDeviceDoubleBuffer();
            std::stringstream step;
            step<<state;
            return std::string("TaskReceive ")+step.str()+std::string(" ")+dbuff.str();
        }

    private:

        enum state_t
        {
            Constructor,
            WaitForReceived,
            RunCopy,
            WaitForFinish,
            WaitForSetSize,
            Finish

        };


        Exchange<TYPE, DIM> *exchange;
        state_t state;
        size_t newBufferSize;
#if(PMACC_ENABLE_GPUDIRECT==1)
        EventTask setSizeEvent;
#endif

    };

} //namespace PMacc


#endif	/* _TASKRECEIVE_HPP */


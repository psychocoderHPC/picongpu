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


#include "eventSystem/tasks/Factory.hpp"

#include "memory/buffers/HostBuffer.hpp"
#include "memory/buffers/DeviceBuffer.hpp"
#include "memory/buffers/Exchange.hpp"

#include "eventSystem/tasks/TaskCopyDeviceToHost.hpp"
#include "eventSystem/tasks/TaskCopyHostToDevice.hpp"
#include "eventSystem/tasks/TaskCopyDeviceToDevice.hpp"
#include "eventSystem/tasks/TaskKernel.hpp"
#include "eventSystem/tasks/TaskReceive.hpp"
#include "eventSystem/tasks/TaskSend.hpp"
#include "eventSystem/tasks/TaskSetValue.hpp"
#include "eventSystem/tasks/TaskSetCurrentSizeOnDevice.hpp"
#include "eventSystem/tasks/TaskSendMPI.hpp"
#include "eventSystem/tasks/TaskReceiveMPI.hpp"
#include "eventSystem/streams/EventStream.hpp"
#include "eventSystem/streams/StreamController.hpp"
#include "eventSystem/tasks/TaskGetCurrentSizeFromDevice.hpp"

namespace PMacc
{

    /**
     * creates a TaskCopyHostToDevice
     * @param src HostBuffer to copy data from
     * @param dst DeviceBuffer to copy data to
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template <typename T_BufferDef>
    inline EventTask Factory::createTaskCopyHostToDevice(HostBuffer<T_BufferDef>& src, DeviceBuffer<T_BufferDef>& dst,
    ITask *registeringTask)
    {

        TaskCopyHostToDevice<T_BufferDef>* task = new TaskCopyHostToDevice<T_BufferDef > (src, dst);

        return startTask(*task, registeringTask);
    }

    /**
     * creates a TaskCopyDeviceToHost
     * @param src DeviceBuffer to copy data from
     * @param dst HostBuffer to copy data to
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template <typename T_BufferDef>
    inline EventTask Factory::createTaskCopyDeviceToHost(DeviceBuffer<T_BufferDef>& src,
    HostBuffer<T_BufferDef>& dst,
    ITask *registeringTask)
    {
        TaskCopyDeviceToHost<T_BufferDef>* task = new TaskCopyDeviceToHost<T_BufferDef > (src, dst);

        return startTask(*task, registeringTask);
    }

    /**
     * creates a TaskCopyDeviceToDevice
     * @param src DeviceBuffer to copy data from
     * @param dst DeviceBuffer to copy data to
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template <typename T_BufferDef>
    inline EventTask Factory::createTaskCopyDeviceToDevice( DeviceBuffer<T_BufferDef>& src, DeviceBuffer<T_BufferDef>& dst,
    ITask *registeringTask)
    {
        TaskCopyDeviceToDevice<T_BufferDef>* task = new TaskCopyDeviceToDevice<T_BufferDef > (src, dst);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a TaskReceive.
     * @param ex Exchange to create new TaskReceive with
     * @param task_out returns the newly created task
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template <typename T_BufferDef>
    inline EventTask Factory::createTaskReceive(Exchange<T_BufferDef> &ex,
    ITask *registeringTask)
    {
        TaskReceive<T_BufferDef>* task = new TaskReceive<T_BufferDef > (ex);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a TaskSend.
     * @param ex Exchange to create new TaskSend with
     * @param task_in TaskReceive to register at new TaskSend
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template <typename T_BufferDef>
    inline EventTask Factory::createTaskSend(Exchange<T_BufferDef> &ex, EventTask &copyEvent,
    ITask *registeringTask)
    {
        TaskSend<T_BufferDef>* task = new TaskSend<T_BufferDef > (ex, copyEvent);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a TaskSendMPI.
     * @param exchange Exchange to create new TaskSendMPI with
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template <typename T_BufferDef>
    inline EventTask Factory::createTaskSendMPI(Exchange<T_BufferDef> *ex,
    ITask *registeringTask)
    {
        TaskSendMPI<T_BufferDef>* task = new TaskSendMPI<T_BufferDef > (ex);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a TaskReceiveMPI.
     * @param ex Exchange to create new TaskReceiveMPI with
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template <typename T_BufferDef>
    inline EventTask Factory::createTaskReceiveMPI(Exchange<T_BufferDef> *ex,
    ITask *registeringTask)
    {
        TaskReceiveMPI<T_BufferDef>* task = new TaskReceiveMPI<T_BufferDef > (ex);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a new TaskSetValue.
     * @param dst destination DeviceBuffer to set value on
     * @param value value to be set in the DeviceBuffer
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template <typename T_BufferDef>
    inline EventTask Factory::createTaskSetValue(DeviceBuffer<T_BufferDef>& dst, const typename T_BufferDef::ValueType& value,
    ITask *registeringTask)
    {

        /* sizeof(TYPE)<256 use fast set method for small data and slow method for big data
         * the rest of 256bytes are reserved for other kernel parameter
         */
        enum
        {
            isSmall = (sizeof (typename T_BufferDef::ValueType) <= 128)
        }; //if we use const variable the compiler create warnings

        TaskSetValue<T_BufferDef, isSmall >* task = new TaskSetValue<T_BufferDef, isSmall > (dst, value);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a new TaskSetCurrentSizeOnDevice.
     * @param dst destination DeviceBuffer to set current size on
     * @param size size to be set on DeviceBuffer
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template <typename T_BufferDef>
    inline EventTask Factory::createTaskSetCurrentSizeOnDevice(DeviceBuffer<T_BufferDef>& dst, size_t size,
    ITask *registeringTask)
    {
        TaskSetCurrentSizeOnDevice<T_BufferDef>* task = new TaskSetCurrentSizeOnDevice<T_BufferDef > (dst, size);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a new TaskGetCurrentSizeFromDevic.
     * @param buffer DeviceBuffer to get current size from
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template <typename T_BufferDef>
    inline EventTask Factory::createTaskGetCurrentSizeFromDevice(DeviceBuffer<T_BufferDef>& buffer,
    ITask *registeringTask)
    {
        TaskGetCurrentSizeFromDevice<T_BufferDef>* task = new TaskGetCurrentSizeFromDevice<T_BufferDef > (buffer);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a new TaskKernel.
     * @param kernelname name of the kernel which should be called
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     * @return the newly created TaskKernel
     */
    inline TaskKernel* Factory::createTaskKernel(std::string kernelname, ITask *registeringTask)
    {
        TaskKernel* task = new TaskKernel(kernelname);

        if (registeringTask != NULL)
            task->addObserver(registeringTask);

        return task;
    }


    inline EventTask Factory::startTask(ITask& task, ITask *registeringTask )
    {
        if (registeringTask != NULL){
            task.addObserver(registeringTask);
        }
        EventTask event(task.getId());

        task.init();
        Environment<>::get().Manager().addTask(&task);
        Environment<>::get().TransactionManager().setTransactionEvent(event);

        return event;
    }


} //namespace PMacc




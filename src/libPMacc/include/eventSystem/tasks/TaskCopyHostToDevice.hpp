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

#ifndef _TASKCOPYHOSTTODEVICE_HPP
#define	_TASKCOPYHOSTTODEVICE_HPP

#include <cuda_runtime_api.h>

#include "eventSystem/EventSystem.hpp"
#include "eventSystem/streams/EventStream.hpp"
#include "eventSystem/tasks/StreamTask.hpp"

namespace PMacc
{

template <typename>
class HostBuffer;
template <typename>
class DeviceBuffer;

template <typename T_BufferDef>
class TaskCopyHostToDeviceBase : public StreamTask
{
public:

    typedef T_BufferDef BufferDef;
    typedef typename BufferDef::ValueType TYPE;
    static const unsigned int DIM = BufferDef::dim;

    TaskCopyHostToDeviceBase(HostBuffer<BufferDef>& src, DeviceBuffer<BufferDef>& dst) :
    StreamTask()
    {
        this->host = &src;
        this->device = &dst;
    }

    virtual ~TaskCopyHostToDeviceBase()
    {
        notify(this->myId, COPYHOST2DEVICE, NULL);
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
        //   __startAtomicTransaction(__getTransactionEvent());
        size_t current_size = host->getCurrentSize();
        DataSpace<DIM> hostCurrentSize = host->getCurrentDataSpace(current_size);
        if (host->is1D() && device->is1D())
            fastCopy(host->getPointer(), device->getPointer(), hostCurrentSize.productOfComponents());
        else
            copy(hostCurrentSize);
        device->setCurrentSize(current_size);
        this->activate();
        //   __setTransactionEvent(__endTransaction());
    }

    std::string toString()
    {
        return "TaskCopyHostToDevice";
    }


protected:

    virtual void copy(DataSpace<DIM> &hostCurrentSize) = 0;

    void fastCopy(TYPE* src, TYPE* dst, size_t size)
    {
        CUDA_CHECK(cudaMemcpyAsync(dst,
                                   src,
                                   size * sizeof (TYPE),
                                   cudaMemcpyHostToDevice,
                                   this->getCudaStream()));
        // std::cout<<"-----------fast H2D"<<std::endl;;
    }


    HostBuffer<BufferDef> *host;
    DeviceBuffer<BufferDef> *device;

};

template <class TYPE, unsigned DIM, typename T_SizeDefinition>
class TaskCopyHostToDevice;

template <class TYPE, typename T_SizeDefinition>
class TaskCopyHostToDevice<TYPE, DIM1, T_SizeDefinition> :
public TaskCopyHostToDeviceBase<BufferDefinition<TYPE, DIM1, T_SizeDefinition> >
{
public:

    typedef BufferDefinition<TYPE, DIM1, T_SizeDefinition> BufferDef;

    TaskCopyHostToDevice(HostBuffer<BufferDef>& src, DeviceBuffer<BufferDef>& dst) :
    TaskCopyHostToDeviceBase<BufferDef>(src, dst)
    {
    }
private:

    virtual void copy(DataSpace<DIM1> &hostCurrentSize)
    {
        CUDA_CHECK(cudaMemcpyAsync(this->device->getPointer(), /*pointer include X offset*/
                                   this->host->getBasePointer(),
                                   hostCurrentSize[0] * sizeof (TYPE), cudaMemcpyHostToDevice,
                                   this->getCudaStream()));
    }
};

template <class TYPE, typename T_SizeDefinition>
class TaskCopyHostToDevice<TYPE, DIM2, T_SizeDefinition> :
public TaskCopyHostToDeviceBase<BufferDefinition<TYPE, DIM2, T_SizeDefinition> >
{
public:
    typedef BufferDefinition<TYPE, DIM2, T_SizeDefinition> BufferDef;

    TaskCopyHostToDevice(HostBuffer<BufferDef>& src, DeviceBuffer<BufferDef>& dst) :
    TaskCopyHostToDeviceBase<BufferDef>(src, dst)
    {
    }
private:

    virtual void copy(DataSpace<DIM2> &hostCurrentSize)
    {
        CUDA_CHECK(cudaMemcpy2DAsync(this->device->getPointer(),
                                     this->device->getPitch(), /*this is pitch*/
                                     this->host->getBasePointer(),
                                     this->host->getDataSpace()[0] * sizeof (TYPE), /*this is pitch*/
                                     hostCurrentSize[0] * sizeof (TYPE),
                                     hostCurrentSize[1],
                                     cudaMemcpyHostToDevice,
                                     this->getCudaStream()));
    }
};

template <class TYPE, typename T_SizeDefinition>
class TaskCopyHostToDevice<TYPE, DIM3, T_SizeDefinition> :
public TaskCopyHostToDeviceBase<BufferDefinition<TYPE, DIM3, T_SizeDefinition> >
{
    public:
    typedef BufferDefinition<TYPE, DIM3, T_SizeDefinition> BufferDef;

    TaskCopyHostToDevice(HostBuffer<BufferDef>& src, DeviceBuffer<BufferDef>& dst) :
        TaskCopyHostToDeviceBase<BufferDef>(src, dst)
    {
    }
    private:

    virtual void copy(DataSpace<DIM3> &hostCurrentSize)
    {
        cudaPitchedPtr hostPtr;
        hostPtr.pitch = this->host->getDataSpace()[0] * sizeof (TYPE);
        hostPtr.ptr = this->host->getBasePointer();
        hostPtr.xsize = this->host->getDataSpace()[0] * sizeof (TYPE);
        hostPtr.ysize = this->host->getDataSpace()[1];

        cudaMemcpy3DParms params;
        params.dstArray = NULL;
        params.dstPos = make_cudaPos(this->device->getOffset()[0] * sizeof (TYPE),
                                     this->device->getOffset()[1],
                                     this->device->getOffset()[2]);
        params.dstPtr = this->device->getCudaPitched();

        params.srcArray = NULL;
        params.srcPos = make_cudaPos(0, 0, 0);
        params.srcPtr = hostPtr;

        params.extent = make_cudaExtent(
                                        hostCurrentSize[0] * sizeof (TYPE),
                                        hostCurrentSize[1],
                                        hostCurrentSize[2]);
        params.kind = cudaMemcpyHostToDevice;

        CUDA_CHECK(cudaMemcpy3DAsync(&params, this->getCudaStream()));
    }
};


} //namespace PMacc


#endif	/* _TASKCOPYHOSTTODEVICE_HPP */


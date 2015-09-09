/**
 * Copyright 2014-2015 Rene Widera, Axel Huebl, Benjamin Worpitz
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "eventSystem/EventSystem.hpp"
#include "eventSystem/tasks/Factory.hpp"
#include "memory/buffers/Buffer.hpp"
#include "memory/buffers/DeviceBuffer.hpp"

#include <alpaka/alpaka.hpp>

#include <cassert>

namespace PMacc
{

/** Implementation of the DeviceBuffer interface for cuda mapped memory
 *
 * For all pmacc tasks and functions this buffer looks like native device buffer
 * but in real it is stored in host memory.
 */
template <class TYPE, unsigned DIM>
class MappedBufferIntern : public DeviceBuffer<TYPE, DIM>
{
public:
    using DataBufHost = alpaka::mem::buf::Buf<
        AlpakaAccDev,
        TYPE,
        alpaka::dim::DimInt<DIM>,
        AlpakaSize>;
    using DataBufDev = alpaka::mem::buf::BufPlainPtrWrapper<
        AlpakaAccDev,
        std::uint32_t,
        alpaka::dim::DimInt<DIM>,
        AlpakaSize>;

    using DataBoxType = typename DeviceBuffer<TYPE, DIM>::DataBoxType;

    MappedBufferIntern(DataSpace<DIM> dataSpace) :
        DeviceBuffer<TYPE, DIM>(dataSpace),
        m_dataBufHost(createData()),
        m_dataBufDev(
                alpaka::mem::getPtrDev(m_dataBufHost, Environment<>::get().DeviceManager().getAccDevice()),
                Environment<>::get().DeviceManager().getAccDevice(),
                dataSpace),
        m_dataViewDev(alpaka::mem::view::createView<DataViewDev>(m_dataBufDev))
    {
        reset(false);
    }

    /**
     * destructor
     */
    virtual ~MappedBufferIntern()
    {
        __startOperation(ITask::TASK_CUDA);
        __startOperation(ITask::TASK_HOST);

        alpaka::mem::buf::unmap(buf, Environment<>::get().DeviceManager().getAccDevice());
    }

    /*! Get unchanged device pointer of memory
     * @return device pointer to memory
     */
    TYPE* getBasePointer()
    {
        __startOperation(ITask::TASK_HOST);
        return alpaka::mem::getPtrDev(m_dataBufHost, Environment<>::get().DeviceManager().getAccDevice());
    }

    /*! Get device pointer of memory
     *
     * This pointer is shifted by the offset, if this buffer points to other
     * existing buffer
     *
     * @return device pointer to memory
     */
    TYPE const * getPointer() const
    {
        return getBasePointer();
    }
    TYPE * getPointer()
    {
        return getBasePointer();
    }

    void setCurrentSize(const size_t size)
    {
        Buffer<TYPE, DIM>::setCurrentSize(size);
    }

    void reset(bool preserveData = true)
    {
        __startOperation(ITask::TASK_HOST);
        this->setCurrentSize(this->getDataSpace().productOfComponents());
        if (!preserveData)
        {
            AlpakaAccStream stream(Environment<>::get().DeviceManager().getAccDevice());
            alpaka::mem::view::set(
                stream,
                m_dataBufHost,
                0,
                this->getDataSpace());
            alpaka::wait::wait(stream);
        }
    }

    void setValue(const TYPE& value)
    {
        __startOperation(ITask::TASK_HOST);
        size_t current_size = this->getCurrentSize();
        for(size_t i = 0; i < current_size; i++)
        {
            alpaka::mem::view::getPtrNative(m_dataBufHost)[i] = value;
        }
    }

    DataBoxType getDataBox()
    {
        __startOperation(ITask::TASK_CUDA);
        return DataBoxType(
            PitchedBox<TYPE, DIM>(
                getBasePointer(),
                getOffset(),
                getDataSpace(),
                getPitch()));
    }

    DataBoxType getHostDataBox()
    {
        __startOperation(ITask::TASK_HOST);
        return DataBoxType(
            PitchedBox<TYPE, DIM>(
                alpaka::mem::view::getPtrNative(m_dataBufHost),
                getOffset(),
                getDataSpace(),
                getPitch()));
    }

    DataSpace<DIM> getOffset() const
    {
        return DataSpace<DIM>();
    }

    bool hasSizeOnAcc() const
    {
        return false;
    }

    SizeBufDev const & getMemBufSizeAcc() const
    {
        throw std::logic_error("getMemBufSizeAcc not implemented by this class");
    }

    SizeBufDev & getMemBufSizeAcc()
    {
        throw std::logic_error("getMemBufSizeAcc not implemented by this class");
    }

    DataViewDev const & getMemBufView() const
    {
        __startOperation(ITask::TASK_HOST);
        __startOperation(ITask::TASK_CUDA);
        return m_dataViewDev;
    }

    DataViewDev & getMemBufView()
    {
        __startOperation(ITask::TASK_HOST);
        __startOperation(ITask::TASK_CUDA);
        return m_dataViewDev;
    }

    size_t getPitch() const
    {
        return alpaka::mem::view::getPitchBytes<DIM-1, size_t>(m_dataBufHost);
    }

    void copyFrom(HostBuffer<TYPE, DIM>& other)
    {
        __startAtomicTransaction(__getTransactionEvent());
        assert(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
        Environment<>::get().Factory().createTaskCopyHostToDevice(other, *this);
        __setTransactionEvent(__endTransaction());
    }

    void copyFrom(DeviceBuffer<TYPE, DIM>& other)
    {
        __startAtomicTransaction(__getTransactionEvent());
        assert(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
        Environment<>::get().Factory().createTaskCopyDeviceToDevice(other, *this);
        __setTransactionEvent(__endTransaction());
    }

private:
    /*! create native array with pitched lines
        */
    DataBufHost createData()
    {
        __startOperation(ITask::TASK_CUDA);

        log<ggLog::MEMORY>("Create mapped device %1%D data: %2% MiB") % DIM % (getDataSpace().productOfComponents() * sizeof(TYPE) / 1024 / 1024 );

        DataBufHost buf(alpaka::mem::buf::alloc<TYPE, AlpakaSize>(
            Environment<>::get().DeviceManager().getHostDevice(),
            getDataSpace()));

        alpaka::mem::buf::map(
            buf,
            Environment<>::get().DeviceManager().getAccDevice());

        return buf;
    }

private:
    DataBufHost m_dataBufHost;
    DataBufDev m_dataBufDev;
    DataViewDev m_dataViewDev;
};
}

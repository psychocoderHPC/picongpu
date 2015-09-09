/**
 * Copyright 2013-2015 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "dimensions/DataSpace.hpp"
#include "eventSystem/tasks/Factory.hpp"
#include "memory/buffers/DeviceBuffer.hpp"
#include "memory/boxes/DataBox.hpp"
#include "algorithms/TypeCast.hpp"

#include <alpaka/alpaka.hpp>

#include <cassert>

namespace PMacc
{

/**
 * Internal device buffer implementation.
 */
template <class TYPE, unsigned DIM>
class DeviceBufferIntern : public DeviceBuffer<TYPE, DIM>
{
public:
    using DataBufDev = alpaka::mem::buf::Buf<
        AlpakaAccDev,
        TYPE,
        alpaka::dim::DimInt<DIM>,
        AlpakaSize>;

    typedef typename DeviceBuffer<TYPE, DIM>::DataBoxType DataBoxType;

    /*! create device buffer
     * @param dataSpace size in any dimension of the grid on the device
     * @param sizeOnDevice memory with the current size of the grid is stored on device
     * @param useVectorAsBase use a vector as base of the array (is not lined pitched)
     *                      if true size on device is atomaticly set to false
     */
    DeviceBufferIntern(DataSpace<DIM> dataSpace, bool _sizeOnDevice = false, bool useVectorAsBase = false) :
        DeviceBuffer<TYPE, DIM>(dataSpace, useVectorAsBase || (DIM==1)),
        m_upDataBufDev(new DataBufDev(useVectorAsBase ? createData1d() : createData())),
        m_dataViewDev(alpaka::mem::view::createView<typename PMacc::DeviceBuffer<TYPE, DIM>::DataViewDev>(*m_upDataBufDev.get()))
    {
        if(_sizeOnDevice && (!useVectorAsBase))
        {
            createSizeOnDevice();
        }
        this->setCurrentSize(this->getDataSpace().productOfComponents());

        reset(false);
    }

    DeviceBufferIntern(DeviceBuffer<TYPE, DIM>& source, DataSpace<DIM> dataSpace, DataSpace<DIM> offset, bool _sizeOnDevice = false) :
        DeviceBuffer<TYPE, DIM>(dataSpace, (DIM==1)),
        m_dataViewDev(
            alpaka::mem::view::createView<typename PMacc::DeviceBuffer<TYPE, DIM>::DataViewDev>(
                source.getMemBufView(),
                PMacc::algorithms::precisionCast::precisionCast<AlpakaSize>(this->getDataSpace()),
                PMacc::algorithms::precisionCast::precisionCast<AlpakaSize>(offset)
            )
        )
    {
        if(_sizeOnDevice)
        {
            createSizeOnDevice();
        }
        setCurrentSize(this->getDataSpace().productOfComponents());
    }

    virtual ~DeviceBufferIntern()
    {
        __startOperation(ITask::TASK_CUDA);
        m_upSizeOnDevice.reset();
    }

    void reset(bool preserveData = true)
    {
        setCurrentSize(this->getDataSpace().productOfComponents());

        __startOperation(ITask::TASK_CUDA);
        if (!preserveData)
        {
            AlpakaAccStream stream(Environment<>::get().DeviceManager().getAccDevice());
            alpaka::mem::view::set(
                stream,
                m_dataViewDev,
                0,
                this->getDataSpace()
            );
            alpaka::wait::wait(stream);
        }
    }

    DataBoxType getDataBox()
    {
        __startOperation(ITask::TASK_CUDA);
        return DataBoxType(
            PitchedBox<TYPE, DIM>(
                getBasePointer(),
                getOffset(),
                this->getDataSpace(),
                getPitch()
            )
        );
    }

    TYPE* getBasePointer()
    {
        __startOperation(ITask::TASK_CUDA);
        return alpaka::mem::view::getPtrNative(alpaka::mem::view::getBuf(m_dataViewDev));
    }

    TYPE const * getPointer() const
    {
        __startOperation(ITask::TASK_CUDA);
        return alpaka::mem::view::getPtrNative(m_dataViewDev);
    }
    TYPE * getPointer()
    {
        __startOperation(ITask::TASK_CUDA);
        return alpaka::mem::view::getPtrNative(m_dataViewDev);
    }

    DataSpace<DIM> getOffset() const
    {
        return DataSpace<DIM>(alpaka::offset::getOffsetsVec(m_dataViewDev));
    }

    bool hasCurrentSizeOnDevice() const
    {
        return m_upSizeOnDevice.get() != nullptr;
    }

    typename PMacc::DeviceBuffer<TYPE, DIM>::SizeBufDev const & getMemBufSizeAcc() const
    {
        __startOperation(ITask::TASK_CUDA);
        if(!m_upSizeOnDevice)
        {
            throw std::runtime_error("Buffer has no size on device!, currentSize is only stored on host side.");
        }
        return *m_upSizeOnDevice.get();
    }
    typename PMacc::DeviceBuffer<TYPE, DIM>::SizeBufDev & getMemBufSizeAcc()
    {
        __startOperation(ITask::TASK_CUDA);
        if(!m_upSizeOnDevice)
        {
            throw std::runtime_error("Buffer has no size on device!, currentSize is only stored on host side.");
        }
        return *m_upSizeOnDevice.get();
    }

    typename PMacc::DeviceBuffer<TYPE, DIM>::DataViewDev const & getMemBufView() const
    {
        __startOperation(ITask::TASK_CUDA);
        return m_dataViewDev;
    }

    typename PMacc::DeviceBuffer<TYPE, DIM>::DataViewDev & getMemBufView()
    {
        __startOperation(ITask::TASK_CUDA);
        return m_dataViewDev;
    }

    /*! Get current size of any dimension
     * @return count of current elements per dimension
     */
    virtual size_t getCurrentSize()
    {
        if(m_upSizeOnDevice)
        {
            __startTransaction(__getTransactionEvent());
            Environment<>::get().Factory().createTaskGetCurrentSizeFromDevice(*this);
            __endTransaction().waitForFinished();
        }

        return this->getSizeHost();
    }

    /**
     * If stream is 0, this function is blocking (we use a kernel to set size).
     * Keep in mind: on Fermi-architecture, kernels in different streams may run at the same time
     * (only used if size is on device).
     */
    void setCurrentSize(const size_t size)
    {
        this->setSizeHost(size);

        if(m_upSizeOnDevice)
        {
            Environment<>::get().Factory().createTaskSetCurrentSizeOnDevice(
                *this, size);
        }
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

    size_t getPitch() const
    {
        return alpaka::mem::view::getPitchBytes<DIM-1>(m_dataViewDev);
    }

    void setValue(const TYPE& value)
    {
        Environment<>::get().Factory().createTaskSetValue(*this, value);
    };

private:
    /*! Creates a ND-buffer with pitch
     */
    DataBufDev createData()
    {
        __startOperation(ITask::TASK_CUDA);

        log<ggLog::MEMORY>("Create device %1%D data: %2% MiB") % DIM % (this->getDataSpace().productOfComponents() * sizeof(TYPE) / 1024 / 1024 );

        return alpaka::mem::buf::alloc<TYPE, AlpakaSize>(
            Environment<>::get().DeviceManager().getAccDevice(),
            PMacc::algorithms::precisionCast::precisionCast<AlpakaSize>(this->getDataSpace())
            );
    }

    /*! Creates a ND-buffer without pitch.
     */
    DataBufDev createData1d()
    {
        __startOperation(ITask::TASK_CUDA);

        log<ggLog::MEMORY>("Create device 1D data: %1% MiB") % (this->getDataSpace().productOfComponents() * sizeof (TYPE) / 1024 / 1024 );

        // \HACK \TODO \FIXME: This allocates the memory twice. One time (possibly with padding) and a second time without padding and deletes the first buffer.
        DataBufDev buf(
            alpaka::mem::buf::alloc<TYPE, AlpakaSize>(
                Environment<>::get().DeviceManager().getAccDevice(),
                PMacc::algorithms::precisionCast::precisionCast<AlpakaSize>(this->getDataSpace())
            ));

        using MemBufFake = alpaka::mem::buf::Buf<
            AlpakaAccDev,
            TYPE,
            alpaka::dim::DimInt<1u>,
            AlpakaSize>;
        MemBufFake fakeBuf(
            alpaka::mem::buf::alloc<TYPE, AlpakaSize>(
                Environment<>::get().DeviceManager().getAccDevice(),
                static_cast<AlpakaSize>(this->getDataSpace().productOfComponents())));

        // Swap the pointers of our buffers.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__) && !defined(PMACC_ACC_CPU)
        buf.m_spMem.swap(fakeBuf.m_spMem);
#else
        TYPE ** bp (const_cast<TYPE **>(&buf.m_spBufCpuImpl->m_pMem));
        TYPE ** fbp (const_cast<TYPE **>(&fakeBuf.m_spBufCpuImpl->m_pMem));
        TYPE * const tmp(*bp);
        *bp = *fbp;
        *fbp = tmp;
#endif

        // Reset the pitch of the original buffer to the correct pitch of the fake buffer.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__) && !defined(PMACC_ACC_CPU)
        buf.m_pitchBytes = this->getDataSpace()[0u] * sizeof(TYPE);
#else
        *const_cast<AlpakaSize *>(&buf.m_spBufCpuImpl->m_pitchBytes) = this->getDataSpace()[0u] * sizeof(TYPE);
#endif

        return buf;
    }

    void createSizeOnDevice()
    {
        __startOperation(ITask::TASK_HOST);
        m_upSizeOnDevice.reset(
            new typename PMacc::DeviceBuffer<TYPE, DIM>::SizeBufDev(
                alpaka::mem::buf::alloc<std::size_t, AlpakaSize>(
                    Environment<>::get().DeviceManager().getAccDevice(),
                    static_cast<AlpakaSize>(1u))));
    }

private:
    std::unique_ptr<typename PMacc::DeviceBuffer<TYPE, DIM>::SizeBufDev> m_upSizeOnDevice;
    std::unique_ptr<DataBufDev> m_upDataBufDev;
    typename PMacc::DeviceBuffer<TYPE, DIM>::DataViewDev m_dataViewDev;
};

} //namespace PMacc

/**
 * Copyright 2013-2015 Rene Widera, Benjamin Worpitz
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

#include "memory/buffers/Buffer.hpp"
#include "eventSystem/tasks/Factory.hpp"
#include "eventSystem/EventSystem.hpp"

#include <alpaka/alpaka.hpp>

#include <cassert>

namespace PMacc
{

/**
 * Internal implementation of the HostBuffer interface.
 */
template <class TYPE, unsigned DIM>
class HostBufferIntern : public HostBuffer<TYPE, DIM>
{
public:
    using DataBufHost = alpaka::mem::buf::Buf<
        AlpakaHost,
        TYPE,
        alpaka::dim::DimInt<DIM>,
        AlpakaSize>;

    using DataBoxType = typename DeviceBuffer<TYPE, DIM>::DataBoxType;

    /**
     * constructor
     * @param dataSpace DataSpace describing the size of the HostBufferIntern to be created
     */
    HostBufferIntern(DataSpace<DIM> dataSpace) :
        HostBuffer<TYPE, DIM>(dataSpace),
        m_upDataBufHost(
            new DataBufHost(
                alpaka::mem::buf::alloc<TYPE, AlpakaSize>(
                    alpaka::dev::cpu::getDev(),
                    PMacc::algorithms::precisionCast::precisionCast<AlpakaSize>(dataSpace))
                )),
        m_dataViewHost(
            alpaka::mem::view::createView<typename PMacc::HostBuffer<TYPE, DIM>::DataViewHost>(
                *m_upDataBufHost.get()
            )
        )
    {
        reset(false);
    }

    HostBufferIntern(HostBufferIntern& source, DataSpace<DIM> dataSpace, DataSpace<DIM> offset = DataSpace<DIM>()) :
        HostBuffer<TYPE, DIM>(dataSpace),
        m_upDataBufHost(),
        m_dataViewHost(
            alpaka::mem::view::createView<typename PMacc::HostBuffer<TYPE, DIM>::DataViewHost>(
                source.getMemBufView(),
                PMacc::algorithms::precisionCast::precisionCast<AlpakaSize>(dataSpace),
                PMacc::algorithms::precisionCast::precisionCast<AlpakaSize>(offset)
            )
        )
    {
        reset(true);
    }

    /**
     * destructor
     */
    virtual ~HostBufferIntern()
    {
        __startOperation(ITask::TASK_HOST);
    }

    /*! Get pointer of memory
        * @return pointer to memory
        */
    TYPE* getBasePointer()
    {
        __startOperation(ITask::TASK_HOST);
        return alpaka::mem::view::getPtrNative(alpaka::mem::view::getBuf(m_dataViewHost));
    }

    TYPE const * getPointer() const
    {
        __startOperation(ITask::TASK_HOST);
        return alpaka::mem::view::getPtrNative(m_dataViewHost);
    }
    TYPE * getPointer()
    {
        __startOperation(ITask::TASK_HOST);
        return alpaka::mem::view::getPtrNative(m_dataViewHost);
    }

    void reset(bool preserveData = true)
    {
        __startOperation(ITask::TASK_HOST);
        this->setCurrentSize(this->getDataSpace().productOfComponents());
        if (!preserveData)
            std::memset(getPointer(), 0, this->getDataSpace().productOfComponents() * sizeof (TYPE));
    }

    void setValue(const TYPE& value)
    {
        __startOperation(ITask::TASK_HOST);
        size_t const currentSize = this->getCurrentSize();
        auto const pPointer(getPointer());
        for (size_t i = 0; i < currentSize; i++)
        {
            pPointer[i] = value;
        }
    }

    DataBoxType getDataBox()
    {
        __startOperation(ITask::TASK_HOST);
        return DataBoxType(PitchedBox<TYPE, DIM>(
            getPointer(),
            DataSpace<DIM>(),
            this->getDataSpace(),
            alpaka::mem::view::getPitchBytes<0u>(m_dataViewHost)));
    }

    typename PMacc::HostBuffer<TYPE, DIM>::DataViewHost const & getMemBufView() const
    {
        __startOperation(ITask::TASK_HOST);
        return m_dataViewHost;
    }
    typename PMacc::HostBuffer<TYPE, DIM>::DataViewHost & getMemBufView()
    {
        __startOperation(ITask::TASK_HOST);
        return m_dataViewHost;
    }

    void copyFrom(DeviceBuffer<TYPE, DIM>& other)
    {
        assert(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
        Environment<>::get().Factory().createTaskCopyDeviceToHost(other, *this);
    }

private:
    std::unique_ptr<DataBufHost> m_upDataBufHost;
    typename PMacc::HostBuffer<TYPE, DIM>::DataViewHost m_dataViewHost;
};
}

/**
 * Copyright 2013-2015 Felix Schmitt, Rene Widera, Benjamin Worpitz
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
#include "eventSystem/events/kernelEvents.hpp"
#include "dimensions/DataSpace.hpp"

namespace PMacc
{
class kernelSetValueOnDeviceMemory
{
public:
    //-----------------------------------------------------------------------------
    //! The kernel.
    //-----------------------------------------------------------------------------
    template<
        typename T_Acc>
    ALPAKA_FN_ACC void operator()(
        T_Acc const &,
        size_t * const pointer,
        size_t const & size) const
    {
        static_assert(
            alpaka::dim::Dim<T_Acc>::value == 1u,
            "kernelSetValueOnDeviceMemory has to be executed in one dimension only!");
        // TODO: Assert that grid block and block thread extents are 1!
        *pointer = size;
    }
};

template <class TYPE, unsigned DIM>
class DeviceBuffer;

template <class TYPE, unsigned DIM>
class TaskSetCurrentSizeOnDevice : public StreamTask
{
public:

    TaskSetCurrentSizeOnDevice(DeviceBuffer<TYPE, DIM>& dst, size_t size) :
    StreamTask(),
    size(size)
    {
        this->destination = & dst;
    }

    virtual ~TaskSetCurrentSizeOnDevice()
    {
        notify(this->myId, SETVALUE, NULL);
    }

    virtual void init()
    {
        setCurrentSize();
    }

    bool executeIntern()
    {
        return isFinished();
    }

    void event(id_t, EventType, IEventData*)
    {
    }

    std::string toString()
    {
        return "TaskSetCurrentSizeOnDevice";
    }

private:

    void setCurrentSize()
    {
        kernelSetValueOnDeviceMemory kernel;
        alpaka::workdiv::WorkDivMembers<alpaka::dim::DimInt<1u>, size_t> workDiv(
            static_cast<std::size_t>(1u),
            static_cast<std::size_t>(1u));
        auto const exec(
            alpaka::exec::create<AlpakaAcc<alpaka::dim::DimInt<1u>>>(
                workDiv,
                kernel,
                alpaka::mem::view::getPtrNative(destination->getMemBufSizeAcc()),
                size));
        alpaka::stream::enqueue(
            this->getEventStream()->getCudaStream(),
            exec);

        activate();
    }

    DeviceBuffer<TYPE, DIM> *destination;
    const size_t size;
};

} //namespace PMacc


/**
 * Copyright 2015 Rene Widera, Benjamin Worpitz
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "particles/MallocMCBuffer.hpp"

namespace picongpu
{
using namespace PMacc;

MallocMCBuffer::MallocMCBuffer( ) :
    upBufHost(),
    hostBufferOffset(0)
{
    /* currently mallocMC has only one heap */
    this->deviceHeapInfo=mallocMC::getHeapLocations()[0];

    upBufWrapperDev.reset(
        new BufWrapperDev(
            reinterpret_cast<char *>(deviceHeapInfo.p),
            Environment<>::get().DeviceManager().getAccDevice(),
            static_cast<AlpakaSize>(deviceHeapInfo.size)));

    Environment<>::get().DataConnector().registerData( *this);
}

MallocMCBuffer::~MallocMCBuffer( )
{
    // alpaka automatically unpins and frees the buffer.
}

void MallocMCBuffer::synchronize( )
{
    /** \todo: we had no abstraction to create a host buffer and a pseudo
     *         device buffer (out of the mallocMC ptr) and copy both with our event
     *         system.
     *         WORKAROUND: use native cuda calls :-(
     */
    if(!upBufHost)
    {
        upBufHost.reset(
            new BufHost(
                alpaka::mem::buf::alloc<char, AlpakaSize>(
                    Environment<>::get().DeviceManager().getHostDevice(),
                    static_cast<AlpakaSize>(deviceHeapInfo.size))));

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
        alpaka::mem::buf::pin(*upBufHost.get());
#endif

        this->hostBufferOffset =
            std::ptrdiff_t(
                alpaka::mem::view::getPtrNative(*upBufWrapperDev.get())
                - alpaka::mem::view::getPtrNative(*upBufHost.get()));
    }
    /* add event system hints */
    __startOperation(ITask::TASK_CUDA);
    __startOperation(ITask::TASK_HOST);

    AlpakaAccStream stream(Environment<>::get().DeviceManager().getAccDevice());
    alpaka::mem::view::copy(
        stream,
        *upBufHost.get(),
        *upBufWrapperDev.get(),
        deviceHeapInfo.size);
    alpaka::wait::wait(stream);
}

} //namespace picongpu

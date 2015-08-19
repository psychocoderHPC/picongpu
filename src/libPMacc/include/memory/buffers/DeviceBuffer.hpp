/**
 * Copyright 2013-2015 Heiko Burau, Rene Widera, Benjamin Worpitz
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

//#include <cuSTL/container/view/View.hpp>
//#include <cuSTL/container/DeviceBuffer.hpp>
#include <math/vector/Int.hpp>
#include <math/vector/Size_t.hpp>
#include <memory/buffers/Buffer.hpp>
#include <types.h>

#include <alpaka/alpaka.hpp>

#include <stdexcept>

namespace PMacc
{
    class EventTask;

    template <class TYPE, unsigned DIM>
    class HostBuffer;

    /**
     * Interface for a DIM-dimensional Buffer of type TYPE on the device.
     *
     * @tparam TYPE datatype of the buffer
     * @tparam DIM dimension of the buffer
     */
    template <class TYPE, unsigned DIM>
    class DeviceBuffer : public Buffer<TYPE, DIM>
    {
    protected:
        using SizeBufDev = alpaka::mem::buf::Buf<
            AlpakaDev,
            std::size_t,
            alpaka::dim::DimInt<1u>,
            std::size_t>;

        using DataViewDev = alpaka::mem::view::View<
            AlpakaDev,
            TYPE,
            alpaka::dim::DimInt<DIM>,
            std::size_t>;

    protected:
        DeviceBuffer(
            DataSpace<DIM> dataSpace,
            bool bData1d) :
            Buffer<TYPE, DIM>(dataSpace, bData1d)
        {}

    public:
        /**
         * Destructor.
         */
        virtual ~DeviceBuffer() = default;

        // \TODO: Remove this method! This is very unclear and should be a CartBuffer constructor!
        //__forceinline__
        /*container::CartBuffer<TYPE, DIM, allocator::DeviceMemAllocator<TYPE, DIM>,
                                copier::D2DCopier<DIM>,
                                assigner::DeviceMemAssigner<DIM> >
        cartBuffer()
        {
            container::DeviceBuffer<TYPE, DIM> result;
            auto & memBufView = getMemBufView();
            result.dataPointer = alpaka::mem::view::getPtrNative(memBufView);
            result._size = (math::Size_t<DIM>)this->getDataSpace();
            if(DIM == 2) result.pitch[0] = alpaka::mem::view::getPitchBytes<1u>(memBufView);
            if(DIM == 3)
            {
                result.pitch[0] = alpaka::mem::view::getPitchBytes<2u>(memBufView);
                result.pitch[1] = alpaka::mem::view::getPitchBytes<1u>(memBufView) * result._size.y();
            }
#ifndef __CUDA_ARCH__
            result.refCount = new int;
#endif
            *result.refCount = 2;
            return result;
        }*/


        /**
         * Returns offset of elements in every dimension.
         *
         * @return count of elements
         */
        virtual DataSpace<DIM> getOffset() const = 0;

        /**
         * Show if current size is stored on device.
         *
         * @return return false if no size is stored on device, true otherwise
         */
        virtual bool hasCurrentSizeOnDevice() const = 0;

        /**
         * Returns memory buffer of current size on accelerator.
         *
         * @return memory buffer of current size on accelerator
         */
        virtual SizeBufDev const & getMemBufSizeAcc() const = 0;
        virtual SizeBufDev & getMemBufSizeAcc() = 0;

        /**
         * Returns the internal alpaka buffer.
         *
         * @return internal alpaka buffer
         */
        virtual DataViewDev const & getMemBufView() const = 0;
        virtual DataViewDev & getMemBufView() = 0;

        /** get line pitch of memory in byte
         *
         * @return size of one line in memory
         */
        virtual size_t getPitch() const = 0;

        /**
         * Copies data from the given HostBuffer to this DeviceBuffer.
         *
         * @param other the HostBuffer to copy from
         */
        virtual void copyFrom(HostBuffer<TYPE, DIM>& other) = 0;

        /**
         * Copies data from the given DeviceBuffer to this DeviceBuffer.
         *
         * @param other the DeviceBuffer to copy from
         */
        virtual void copyFrom(DeviceBuffer<TYPE, DIM>& other) = 0;

    };

} //namespace PMacc

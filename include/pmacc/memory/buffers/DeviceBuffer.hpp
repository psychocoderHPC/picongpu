/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/assert.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/buffers/Buffer.hpp"
#include "pmacc/types.hpp"

#include <memory>

namespace pmacc
{
    class EventTask;

    template<class T_Type, unsigned T_dim>
    class HostBuffer;

    template<class T_Type, unsigned T_dim>
    class DeviceBuffer;

    template<class T_Type, unsigned T_dim>
    class Buffer;

    /** T_dim-dimensional Buffer of type T_Type on the device.
     *
     * @tparam T_Type datatype of the buffer
     * @tparam T_dim dimension of the buffer
     */
    template<class T_Type, unsigned T_dim>
    class DeviceBuffer
        : public Buffer<T_Type, T_dim>
        , public std::enable_shared_from_this<DeviceBuffer<T_Type, T_dim>>
    {
        using BufferType = ::alpaka::Buf<AccDev, T_Type, AlpakaDim<DIM1>, IdxType>;
        using ViewType = alpaka::ViewPlainPtr<AccDev, T_Type, AlpakaDim<T_dim>, IdxType>;
        using CurrentSizeBufferDevice = ::alpaka::Buf<AccDev, size_t, AlpakaDim<DIM1>, IdxType>;

    public:
        using DataBoxType = typename Buffer<T_Type, T_dim>::DataBoxType;
        std::shared_ptr<BufferType> devBuffer;
        std::shared_ptr<ViewType> view;
        std::shared_ptr<CurrentSizeBufferDevice> currentSizeBufferDevice;

        using BufferType1D = ::alpaka::ViewPlainPtr<AccDev, T_Type, AlpakaDim<DIM1>, IdxType>;

        BufferType1D as1DBuffer()
        {
            auto currentSize = this->getCurrentSize();
            eventSystem::startOperation(ITask::TASK_DEVICE);
            return BufferType1D(
                alpaka::getPtrNative(*view),
                alpaka::getDev(*devBuffer),
                DataSpace<DIM1>(currentSize).toAlpakaVec());
        }

        BufferType1D as1DBufferNElem(size_t const numElements)
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            return BufferType1D(
                alpaka::getPtrNative(*view),
                alpaka::getDev(*devBuffer),
                DataSpace<DIM1>(numElements).toAlpakaVec());
        }

        ViewType& getAlpakaView() const
        {
            return *view;
        }

        /** Create device buffer
         *
         * Allocate new memory on the device.
         *
         * @param size extent for each dimension (in elements)
         * @param sizeOnDevice memory with the current size of the grid is stored on device
         */
        DeviceBuffer(DataSpace<T_dim> const& size, bool sizeOnDevice = false)
            : Buffer<T_Type, T_dim>(size)
            , devBuffer(std::make_shared<BufferType>(alpaka::allocBuf<T_Type, IdxType>(
                  manager::Device<AccDev>::get().current(),
                  DataSpace<DIM1>(size.productOfComponents()).toAlpakaVec())))
        {
            DataSpace<T_dim> pitchInBytes;
            pitchInBytes.x() = sizeof(T_Type);
            for(uint32_t d = 1u; d < T_dim; ++d)
                pitchInBytes[d] = pitchInBytes[d - 1u] * size[d - 1u];
            view = std::make_shared<ViewType>(
                alpaka::getPtrNative(*devBuffer),
                alpaka::getDev(*devBuffer),
                size.toAlpakaVec(),
                pitchInBytes.toAlpakaVec());

            if(sizeOnDevice)
                currentSizeBufferDevice = std::make_shared<CurrentSizeBufferDevice>(alpaka::allocBuf<size_t, IdxType>(
                    manager::Device<AccDev>::get().current(),
                    DataSpace<DIM1>(1).toAlpakaVec()));
            this->setCurrentSize(size.productOfComponents());
            this->data1D = true;
            reset(false);
        }

        /** Create a shallow copy of the given source buffer
         *
         * The resulting buffer is effectively a subview to the source buffer.
         *
         * @param source source device buffer
         * @param size extent for each dimension (in elements)
         * @param offset extra offset in the source buffer
         * @param sizeOnDevice memory with the current size of the grid is stored on device
         */
        DeviceBuffer(
            DeviceBuffer<T_Type, T_dim>& source,
            DataSpace<T_dim> size,
            DataSpace<T_dim> offset,
            bool sizeOnDevice = false)
            : Buffer<T_Type, T_dim>(size)
            , devBuffer(source.devBuffer)
        {
            auto subView = createSubView(*source.view, size.toAlpakaVec(), offset.toAlpakaVec());
            view = std::make_shared<ViewType>(
                alpaka::getPtrNative(subView),
                alpaka::getDev(subView),
                alpaka::getExtents(subView),
                alpaka::getPitchesInBytes(subView));
            if(sizeOnDevice)
                currentSizeBufferDevice = std::make_shared<CurrentSizeBufferDevice>(alpaka::allocBuf<size_t, IdxType>(
                    manager::Device<AccDev>::get().current(),
                    DataSpace<DIM1>(1).toAlpakaVec()));
            this->setCurrentSize(size.productOfComponents());
            this->data1D = false || T_dim == DIM1;
            reset(true);
        }

        ~DeviceBuffer() override
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
        }

        void reset(bool preserveData = true) override
        {
            this->setCurrentSize(Buffer<T_Type, T_dim>::getDataSpace().productOfComponents());

            eventSystem::startOperation(ITask::TASK_DEVICE);
            if(!preserveData)
            {
                // Using Array is a workaround for types without default constructor
                memory::Array<uint8_t, sizeof(T_Type)> tmp(uint8_t{0});
                // use first element to avoid issue because Array is aligned (sizeof can be larger than component type)
                setValue(*reinterpret_cast<T_Type*>(tmp.data()));
            }
        }

        DataBoxType getDataBox() override
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            auto pitchBytes = DataSpace<T_dim>(getPitchesInBytes(*view));
            return DataBoxType(PitchedBox<T_Type, T_dim>(alpaka::getPtrNative(*view), pitchBytes));
        }

        /**
         * Show if current size is stored on device.
         *
         * @return return false if no size is stored on device, true otherwise
         */
        bool hasCurrentSizeOnDevice() const
        {
            return currentSizeBufferDevice != nullptr;
        }

        /**
         * Returns pointer to current size on device.
         *
         * @return pointer which point to device memory of current size
         */
        auto getCurrentSizeOnDeviceBuffer()
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            if(!hasCurrentSizeOnDevice())
            {
                throw std::runtime_error("Buffer has no size on device!, currentSize is only stored on host side.");
            }
            return currentSizeBufferDevice;
        }

        /*! Get current size of any dimension
         * @return count of current elements per dimension
         */
        size_t getCurrentSize() override
        {
            if(hasCurrentSizeOnDevice())
            {
                eventSystem::startTransaction(eventSystem::getTransactionEvent());
                Environment<>::get().Factory().createTaskGetCurrentSizeFromDevice(*this);
                eventSystem::endTransaction().waitForFinished();
            }

            return Buffer<T_Type, T_dim>::getCurrentSize();
        }

        /**
         * Sets current size of any dimension.
         *
         * If stream is 0, this function is blocking (we use a kernel to set size).
         * Keep in mind: on Fermi-architecture, kernels in different streams may run at the same time
         * (only used if size is on device).
         *
         * @param size count of elements per dimension
         */
        void setCurrentSize(const size_t newSize) override
        {
            Buffer<T_Type, T_dim>::setCurrentSize(newSize);

            if(hasCurrentSizeOnDevice())
            {
                Environment<>::get().Factory().createTaskSetCurrentSizeOnDevice(*this, newSize);
            }
        }

        /**
         * Copies data from the given HostBuffer to this DeviceBuffer.
         *
         * @param other the HostBuffer to copy from
         */
        void copyFrom(HostBuffer<T_Type, T_dim>& other)
        {
            PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
            Environment<>::get().Factory().createTaskCopy(other, *this);
        }

        /**
         * Copies data from the given DeviceBuffer to this DeviceBuffer.
         *
         * @param other the DeviceBuffer to copy from
         */
        void copyFrom(DeviceBuffer<T_Type, T_dim>& other)
        {
            PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
            Environment<>::get().Factory().createTaskCopy(other, *this);
        }

        void setValue(T_Type const& value) override
        {
            Environment<>::get().Factory().createTaskSetValue(*this, value);
        };

        auto getCurrentSizeDeviceSideBuffer()
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            return currentSizeBufferDevice;
        }

        typename Buffer<T_Type, T_dim>::CPtr getCPtr(bool send) final
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            if(send)
                return {alpaka::getPtrNative(*view), this->getCurrentSize()};
            else
                return {alpaka::getPtrNative(*view), this->getDataSpace().productOfComponents()};
        }
    };

    /** Factory for a new heap-allocated DeviceBuffer buffer object that is a deep copy of the given device
     * buffer
     *
     * @tparam T_Type value type
     * @tparam T_dim index dimensionality
     *
     * @param source source device buffer
     */
    template<class T_Type, unsigned T_dim>
    HINLINE std::unique_ptr<DeviceBuffer<T_Type, T_dim>> makeDeepCopy(DeviceBuffer<T_Type, T_dim>& source)
    {
        // We have to call this constructor to allocate a new data storage and not shallow-copy the source
        auto result = std::make_unique<DeviceBuffer<T_Type, T_dim>>(source.getDataSpace());
        result->copyFrom(source);
        // Wait for copy to finish, so that the resulting object is safe to use after return
        eventSystem::getTransactionEvent().waitForFinished();
        return result;
    }

} // namespace pmacc

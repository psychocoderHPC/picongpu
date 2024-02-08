/* Copyright 2013-2023 Rene Widera, Benjamin Worpitz,
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

#include "pmacc/acc.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/cuplaHelper/Device.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/eventSystem/eventSystem.hpp"
#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/memory/boxes/DataBoxDim1Access.hpp"
#include "pmacc/memory/buffers/Buffer.hpp"

namespace pmacc
{
    class EventTask;

    template<class TYPE, unsigned DIM>
    class DeviceBuffer;

    template<class TYPE, unsigned DIM>
    class Buffer;

    /** DIM-dimensional Buffer of type TYPE on the host
     *
     * @tparam T_Type datatype for buffer data
     * @tparam T_dim dimension of the buffer
     */
    template<typename T_Type, uint32_t T_dim>
    class HostBuffer : public Buffer<T_Type, T_dim>
    {
        using BufferType = ::alpaka::Buf<AccHost, T_Type, AlpakaDim<DIM1>, IdxType>;
        using ViewType = alpaka::ViewPlainPtr<AccHost, T_Type, AlpakaDim<T_dim>, IdxType>;

    public:
        using DataBoxType = typename Buffer<T_Type, T_dim>::DataBoxType;
        std::shared_ptr<BufferType> hostBuffer;
        std::shared_ptr<ViewType> view;

        using BufferType1D = ::alpaka::ViewPlainPtr<AccHost, T_Type, AlpakaDim<DIM1>, IdxType>;

        BufferType1D as1DBuffer()
        {
            auto currentSize = this->getCurrentSize();
            eventSystem::startOperation(ITask::TASK_DEVICE);
            return BufferType1D(
                alpaka::getPtrNative(*view),
                alpaka::getDev(*hostBuffer),
                DataSpace<DIM1>(currentSize).toAlpakaVec());
        }

        ViewType& getAlpakaView() const
        {
            return *view;
        }

        /** constructor
         *
         * @param size extent for each dimension (in elements)
         */
        HostBuffer(DataSpace<T_dim> size)
            : Buffer<T_Type, T_dim>(size)
            , hostBuffer(std::make_shared<BufferType>(alpaka::allocMappedBufIfSupported<T_Type, IdxType>(
                  manager::Device<AccHost>::get().current(),
                  manager::Device<AccDev>::get().getPlatform(),
                  DataSpace<DIM1>(size.productOfComponents()).toAlpakaVec())))
        {
            DataSpace<T_dim> pitchInBytes;
            pitchInBytes.x() = sizeof(T_Type);
            for(uint32_t d = 1u; d < T_dim; ++d)
                pitchInBytes[d] = pitchInBytes[d - 1u] * size[d - 1u];
            view = std::make_shared<ViewType>(
                alpaka::getPtrNative(*hostBuffer),
                alpaka::getDev(*hostBuffer),
                size.toAlpakaVec(),
                pitchInBytes.toAlpakaVec());
            reset(false);
        }

        HostBuffer(HostBuffer& source, DataSpace<T_dim> size, DataSpace<T_dim> offset = DataSpace<T_dim>())
            : Buffer<T_Type, T_dim>(size)
            , hostBuffer(source->hostBuffer)
        {
            auto subView = createSubView(*source.view, size.toAlpakaVec(), offset.toAlpakaVec());
            view = std::make_shared<ViewType>(
                alpaka::getPtrNative(subView),
                alpaka::getDev(subView),
                alpaka::getExtents(subView),
                alpaka::getPitchesInBytes(subView));
            reset(true);
        }

        /**
         * destructor
         */
        ~HostBuffer() override
        {
            eventSystem::startOperation(ITask::TASK_HOST);
        }

        /**
         * Copies the data from the given DeviceBuffer to this HostBuffer.
         *
         * @param other DeviceBuffer to copy data from
         */
        void copyFrom(DeviceBuffer<T_Type, T_dim>& other)
        {
            PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
            Environment<>::get().Factory().createTaskCopy(other, *this);
        }

        void reset(bool preserveData = true)
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            this->setCurrentSize(this->getDataSpace().productOfComponents());
            if(!preserveData)
            {
                /* if it is a pointer out of other memory we can not assume that
                 * that the physical memory is contiguous
                 */
                if(hostBuffer && alpaka::getPtrNative(*hostBuffer) == alpaka::getPtrNative(*view))
                    memset(
                        reinterpret_cast<void*>(alpaka::getPtrNative(*view)),
                        0,
                        this->getDataSpace().productOfComponents() * sizeof(T_Type));
                else
                {
                    // Using Array is a workaround for types without default constructor
                    memory::Array<uint8_t, sizeof(T_Type)> tmp(uint8_t{0});
                    // use first element to avoid issue because Array is aligned (sizeof can be larger than component
                    // type)
                    setValue(*reinterpret_cast<T_Type*>(tmp.data()));
                }
            }
        }

        void setValue(const T_Type& value)
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            auto current_size = static_cast<int64_t>(this->getCurrentSize());
            auto memBox = getDataBox();
            using D1Box = DataBoxDim1Access<DataBoxType>;
            D1Box d1Box(memBox, this->getDataSpace());
#pragma omp parallel for
            for(int64_t i = 0; i < current_size; i++)
            {
                d1Box[i] = value;
            }
        }

        DataBoxType getDataBox()
        {
            auto pitchBytes = DataSpace<T_dim>(getPitchesInBytes(*view));
            eventSystem::startOperation(ITask::TASK_HOST);
            return DataBoxType(PitchedBox<T_Type, T_dim>(alpaka::getPtrNative(*view), pitchBytes));
        }

        typename Buffer<T_Type, T_dim>::CPtr getCPtr(bool send) final
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            if(send)
                return {alpaka::getPtrNative(*view), this->getCurrentSize()};
            else
                return {alpaka::getPtrNative(*view), this->getDataSpace().productOfComponents()};
        }
    };

} // namespace pmacc

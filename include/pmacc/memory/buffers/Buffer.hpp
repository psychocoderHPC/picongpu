/* Copyright 2013-2023 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "pmacc/Environment.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/PitchedBox.hpp"
#include "pmacc/types.hpp"

#include <limits>

namespace pmacc
{
    /**
     * Minimal function description of a buffer,
     *
     * @tparam T_Type data type stored in the buffer
     * @tparam T_dim dimension of the buffer (1-3)
     */
    template<class T_Type, unsigned T_dim>
    class Buffer
    {
    protected:
        using CurrentSizeBuffer = ::alpaka::Buf<AccHost, size_t, AlpakaDim<DIM1>, IdxType>;
        std::shared_ptr<CurrentSizeBuffer> currentSizeBuffer;
        DataSpace<T_dim> capasity;

    public:
        using DataBoxType = DataBox<PitchedBox<T_Type, T_dim>>;

        /** constructor
         *
         * @param size extent for each dimension (in elements)
         *             if the buffer is a view to an existing buffer the size
         *             can be less than `physicalMemorySize`
         */
        Buffer(DataSpace<T_dim> size)
            : currentSizeBuffer(std::make_shared<CurrentSizeBuffer>(alpaka::allocMappedBufIfSupported<size_t, IdxType>(
                manager::Device<AccHost>::get().current(),
                manager::Device<AccDev>::get().getPlatform(),
                DataSpace<DIM1>(1).toAlpakaVec())))
            , capasity(size)
            , data1D(true)
        {
        }

        /**
         * destructor
         */
        virtual ~Buffer()
        {
            eventSystem::startOperation(ITask::TASK_HOST);
        }

        DataSpace<T_dim> getDataSpace() const
        {
            return capasity;
        }

        /** Returns host pointer of current size storage
         *
         * @return pointer to stored value on host side
         */
        auto getCurrentSizeHostSideBuffer()
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            return currentSizeBuffer;
        }

        virtual size_t getCurrentSize()
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            return alpaka::getPtrNative(*currentSizeBuffer)[0];
        }

        virtual void setCurrentSize(size_t const newSize)
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            PMACC_ASSERT(static_cast<size_t>(newSize) <= static_cast<size_t>(getDataSpace().productOfComponents()));
            alpaka::getPtrNative(*currentSizeBuffer)[0] = newSize;
        }

        virtual DataSpace<T_dim> getCurrentDataSpace()
        {
            return getCurrentDataSpace(getCurrentSize());
        }

        /*! Spread of memory per dimension which is currently used
         * @return if DIM == DIM1 than return count of elements (x-direction)
         * if DIM == DIM2 than return how many lines (y-direction) of memory is used
         * if DIM == DIM3 than return how many slides (z-direction) of memory is used
         */
        virtual DataSpace<T_dim> getCurrentDataSpace(size_t currentSize)
        {
            DataSpace<T_dim> tmp;
            auto current_size = static_cast<int64_t>(currentSize);

            //!\todo: current size can be changed if it is a DeviceBuffer and current size is on device
            // call first get current size (but const not allow this)

            if constexpr(T_dim == DIM1)
            {
                tmp[0] = current_size;
            }
            if constexpr(T_dim == DIM2)
            {
                if(current_size <= capasity[0])
                {
                    tmp[0] = current_size;
                    tmp[1] = 1;
                }
                else
                {
                    tmp[0] = capasity[0];
                    tmp[1] = (current_size + capasity[0] - 1) / capasity[0];
                }
            }
            if constexpr(T_dim == DIM3)
            {
                if(current_size <= capasity[0])
                {
                    tmp[0] = current_size;
                    tmp[1] = 1;
                    tmp[2] = 1;
                }
                else if(current_size <= (capasity[0] * capasity[1]))
                {
                    tmp[0] = capasity[0];
                    tmp[1] = (current_size + capasity[0] - 1) / capasity[0];
                    tmp[2] = 1;
                }
                else
                {
                    tmp[0] = capasity[0];
                    tmp[1] = capasity[1];
                    tmp[2] = (current_size + (capasity[0] * capasity[1]) - 1) / (capasity[0] * capasity[1]);
                }
            }

            return tmp;
        }

        virtual void reset(bool preserveData = false) = 0;

        virtual void setValue(T_Type const& value) = 0;

        virtual DataBox<PitchedBox<T_Type, T_dim>> getDataBox() = 0;

        inline bool is1D()
        {
            return data1D;
        }

        struct CPtr
        {
            T_Type* ptr;
            size_t size;

            size_t sizeInBytes() const
            {
                return size * sizeof(T_Type);
            }

            char* asCharPtr() const
            {
                return reinterpret_cast<char*>(ptr);
            }
        };

        virtual CPtr getCPtr(bool send) = 0;

    protected:
        /*! Check if my DataSpace is greater than other.
         * @param other other DataSpace
         * @return true if my DataSpace (one dimension) is greater than other, false otherwise
         */
        virtual bool isMyDataSpaceGreaterThan(DataSpace<T_dim> other)
        {
            return !other.isOneDimensionGreaterThan(getDataSpace());
        }

        bool data1D = true;
    };

} // namespace pmacc

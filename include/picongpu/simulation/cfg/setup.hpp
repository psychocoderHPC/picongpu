/* Copyright 2023 Rene Widera
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

#include "picongpu/simulation/cfg/Simulation.hpp"
#include "picongpu/simulation/cfg/Unitsystem.hpp"
#include "pmacc/eventSystem/eventSystem.hpp"

#include <pmacc/eventSystem/tasks/ITask.hpp>
#include <pmacc/memory/Array.hpp>

#include <cupla.hpp>

namespace picongpu
{
#if defined(__CUDACC__) || BOOST_COMP_HIP
    namespace setupOnDevice
    {
        ALPAKA_STATIC_ACC_MEM_CONSTANT pmacc::memory::Array<simulation::cfg::Simulation, 1> configArray;

        DINLINE auto& getSetup()
        {
            return configArray[0];
        }
    } // namespace setupOnDevice
#endif
    namespace setupOnHost
    {
        inline simulation::cfg::Simulation& getSetup()
        {
            static simulation::cfg::Simulation setup;
            return setup;
        }
    } // namespace setupOnHost


    inline void upload(picongpu::simulation::cfg::Simulation& sim)
    {
#if defined(__CUDACC__) || BOOST_COMP_HIP
        eventSystem::startOperation(pmacc::ITask::TASK_DEVICE);

        const ::alpaka::Vec<cupla::AlpakaDim<1u>, cupla::MemSizeType> extent(1u);

        auto viewSetupOnDevice = alpaka::createStaticDevMemView(
            &setupOnDevice::configArray[0],
            cupla::manager::Device<cupla::AccDev>::get().current(),
            extent);

        auto hostDevice = cupla::manager::Device<cupla::AccHost>::get().current();
        auto viewSetupOnHost = alpaka::createView(hostDevice, &sim, extent);
        auto queue = cupla::manager::Stream<cupla::AccDev, cupla::AccStream>::get().stream(
            eventSystem::getEventStream(pmacc::ITask::TASK_DEVICE)->getCudaStream());
        alpaka::memcpy(queue, viewSetupOnDevice, viewSetupOnHost, extent);
        alpaka::wait(queue);

        std::cout << "data copied to device" << std::endl;
#endif
    }

/* select namespace depending on __CUDA_ARCH__ compiler flag*/
#if(CUPLA_DEVICE_COMPILE == 1 && /* we are on gpu ... and not using an offloading backend: */                         \
    !(defined ALPAKA_ACC_ANY_BT_OMP5_ENABLED || defined ALPAKA_ACC_ANY_BT_OACC_ENABLED))

    using namespace setupOnDevice;
    DINLINE auto const& setup(unit::Pic const)
    {
        return setupOnDevice::getSetup().pic;
    }

    DINLINE auto const& setup(unit::Si const)
    {
        return setupOnDevice::getSetup().si;
    }

    DINLINE auto const& setup()
    {
        return setup(unit::Pic{});
    }
#else
    using namespace setupOnHost;
    inline auto const& setup(unit::Pic const)
    {
        return setupOnHost::getSetup().pic;
    }

    inline auto const& setup(unit::Si const)
    {
        return setupOnHost::getSetup().si;
    }

    inline auto const& setup()
    {
        return setup(unit::Pic{});
    }
#endif

} // namespace picongpu

/**
 * Copyright 2014-2015 Felix Schmitt, Conrad Schumann, Benjamin Worpitz
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

#include "types.h"

#include <memory>

namespace PMacc
{
    // \TODO: Move to own header!
    class DeviceManager
    {
    public:
        void init(std::size_t uiIdx)
        {
            m_hostDev.reset(
                new AlpakaHostDev(alpaka::dev::cpu::getDev()));

            auto const uiNumDevices(alpaka::dev::DevMan<AlpakaAccDev>::getDevCount());

            // Beginning from the device given by the index, try if they are usable.
            for(std::size_t iDeviceOffset(0); iDeviceOffset < uiNumDevices; ++iDeviceOffset)
            {
                std::size_t const iDevice((uiIdx + iDeviceOffset) % uiNumDevices);

                try
                {
                    m_accDev.reset(
                        new AlpakaAccDev(alpaka::dev::DevMan<AlpakaAccDev>::getDevByIdx(iDevice)));
                    return;
                }
                catch(...)
                {}
            }

            // If we came until here, none of the devices was usable.
            std::stringstream ssErr;
            ssErr << "Unable to return device handle for device " << uiIdx << " because none of the " << uiNumDevices << " devices is accessible!";
            throw std::runtime_error(ssErr.str());
        }

        AlpakaAccDev const & getAccDevice() const
        {
            return *m_accDev.get();
        }

        AlpakaAccDev & getAccDevice()
        {
            return *m_accDev.get();
        }

        AlpakaHostDev const & getHostDevice() const
        {
            return *m_hostDev.get();
        }

        AlpakaHostDev & getHostDevice()
        {
            return *m_hostDev.get();
        }

        static DeviceManager& getInstance()
        {
            static DeviceManager instance;
            return instance;
        }

    private:
        std::unique_ptr<AlpakaAccDev> m_accDev;
        std::unique_ptr<AlpakaHostDev> m_hostDev;
    };
}

#include "eventSystem/EventSystem.hpp"
#include "particles/tasks/ParticleFactory.hpp"

#include "mappings/simulation/GridController.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "mappings/simulation/EnvironmentController.hpp"
#include "eventSystem/streams/StreamController.hpp"
#include "dataManagement/DataConnector.hpp"
#include "pluginSystem/PluginConnector.hpp"
#include "nvidia/memory/MemoryInfo.hpp"
#include "mappings/simulation/Filesystem.hpp"


namespace PMacc
{

/**
 * Global Environment singleton for Picongpu
 */

template <unsigned DIM = DIM1>
class Environment
{
public:

    PMacc::GridController<DIM>& GridController()
    {
        return PMacc::GridController<DIM>::getInstance();
    }

    PMacc::DeviceManager& DeviceManager()
    {
        return PMacc::DeviceManager::getInstance();
    }

    PMacc::StreamController& StreamController()
    {
        return StreamController::getInstance();
    }

    PMacc::Manager& Manager()
    {
        return Manager::getInstance();
    }

    PMacc::TransactionManager& TransactionManager() const
    {
        return TransactionManager::getInstance();
    }

    PMacc::SubGrid<DIM>& SubGrid()
    {
        return PMacc::SubGrid<DIM>::getInstance();
    }

    PMacc::EnvironmentController& EnvironmentController()
    {
        return EnvironmentController::getInstance();
    }

    PMacc::Factory& Factory()
    {
        return Factory::getInstance();
    }

    PMacc::ParticleFactory& ParticleFactory()
    {
        return ParticleFactory::getInstance();
    }

    PMacc::DataConnector& DataConnector()
    {
        return DataConnector::getInstance();
    }

    PMacc::PluginConnector& PluginConnector()
    {
        return PluginConnector::getInstance();
    }

    nvidia::memory::MemoryInfo& EnvMemoryInfo()
    {
        return nvidia::memory::MemoryInfo::getInstance();
    }

    PMacc::Filesystem<DIM>& Filesystem()
    {
        return PMacc::Filesystem<DIM>::getInstance();
    }

    static Environment<DIM>& get()
    {
        static Environment<DIM> instance;
        return instance;
    }

    void initDevices(DataSpace<DIM> devices, DataSpace<DIM> periodic)
    {
        PMacc::GridController<DIM>::getInstance().init(devices, periodic);

        PMacc::Filesystem<DIM>::getInstance();

        PMacc::DeviceManager::getInstance().init(static_cast<std::size_t>(PMacc::GridController<DIM>::getInstance().getHostRank()));

        PMacc::StreamController::getInstance().activate(PMacc::DeviceManager::getInstance().getAccDevice());

        nvidia::memory::MemoryInfo::getInstance().activate(PMacc::DeviceManager::getInstance().getAccDevice());

        PMacc::TransactionManager::getInstance();

    }

    void initGrids(DataSpace<DIM> gridSizeGlobal, DataSpace<DIM> gridSizeLocal, DataSpace<DIM> gridOffset)
    {
        PMacc::SubGrid<DIM>::getInstance().init(gridSizeLocal, gridSizeGlobal, gridOffset);

        PMacc::EnvironmentController::getInstance();

        PMacc::DataConnector::getInstance();

        PMacc::PluginConnector::getInstance();

        nvidia::memory::MemoryInfo::getInstance();

    }

    void finalize()
    {
    }

private:

    Environment()
    {
    }

    Environment(const Environment&);

    Environment& operator=(const Environment&);
};

#define __startTransaction(...) (PMacc::Environment<>::get().TransactionManager().startTransaction(__VA_ARGS__))
#define __startAtomicTransaction(...) (PMacc::Environment<>::get().TransactionManager().startAtomicTransaction(__VA_ARGS__))
#define __endTransaction() (PMacc::Environment<>::get().TransactionManager().endTransaction())
#define __startOperation(opType) (PMacc::Environment<>::get().TransactionManager().startOperation(opType))
#define __getTransactionEvent() (PMacc::Environment<>::get().TransactionManager().getTransactionEvent())
#define __setTransactionEvent(event) (PMacc::Environment<>::get().TransactionManager().setTransactionEvent((event)))

}

#include "particles/tasks/ParticleFactory.tpp"

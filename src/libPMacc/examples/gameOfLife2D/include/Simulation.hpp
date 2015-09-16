/**
 * Copyright 2013-2014 Rene Widera, Maximilian Knespel, Benjamin Worpitz
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

#include "types.hpp"
#include "dimensions/DataSpace.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "mappings/kernel/MappingDescription.hpp"
#include "memory/buffers/GridBuffer.hpp"
#include "memory/dataTypes/Mask.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "Evolution.hpp"
#include "eventSystem/EventSystem.hpp"

#include "GatherSlice.hpp"
#include "traits/NumberOfExchanges.hpp"
#include "GrayImgWriter.hpp"

#include <string>
#include <memory>

namespace gol
{

//#############################################################################
//!
//#############################################################################
class Simulation
{
private:
    // Arbitrarily chosen SuperCellSize!
#if ALPAKA_ACC_GPU_CUDA_ENABLE
    using MappingDesc = PMacc::MappingDescription<DIM2, PMacc::math::CT::Int<16, 16>>;
#else
    using MappingDesc = PMacc::MappingDescription<DIM2, PMacc::math::CT::Int<1, 1>>;
#endif

public:
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    Simulation(
        uint32_t rule,
        std::size_t steps,
        PMacc::DataSpace<DIM2> gridSize,
        PMacc::DataSpace<DIM2> devices,
        PMacc::DataSpace<DIM2> periodic) :
            evo(rule),
            steps(steps),
            gridSize(gridSize),
            isMaster(false),
            buff1(nullptr),
            buff2(nullptr)
    {
        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

        // - First this initializes the GridController with number of 'devices' and 'periodic'ity.
        //   The init-routine will then create and manage the MPI processes and communication group and topology.
        // - Second the devices will be allocated to the corresponding Host MPI processes where hostRank == deviceNumber,
        //   if the device is not marked to be used exclusively by another process.
        //   This affects: cudaMalloc, cudaKernelLaunch, ...
        // - Then the EventStream Controller is activated and one stream is added.
        //   It's basically a List of streams used to parallelize Memory transfers and calculations.
        // - Initialize TransactionManager
        PMacc::Environment<DIM2>::get().initDevices(
            devices,
            periodic);

        // Now we have allocated every node to a grid position in the GC.
        // We use that grid position to allocate every node to a position in the physic grid.
        // Using the localGridSize = the number of cells per node = number of cells / nodes,
        // we can get the position of the current node as an offset in numbers of cells
        PMacc::GridController<DIM2> & gc(
            PMacc::Environment<DIM2>::get().GridController());
        PMacc::DataSpace<DIM2> localGridSize(
            gridSize / devices);

        // - This forwards arguments to SubGrid.init()
        // - Create Singletons: EnvironmentController, DataConnector, PluginConnector, nvidia::memory::MemoryInfo
        PMacc::Environment<DIM2>::get().initGrids(
            gridSize,
            localGridSize,
            gc.getPosition() * localGridSize);
    }

    virtual ~Simulation()
    {
    }

    void finalize()
    {
        gather.finalize();
        buff1.reset();
        buff2.reset();
    }

    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    void init()
    {
        /* subGrid holds global and*
         * local SimulationSize and where the local SimArea is in the greater *
         * scheme using Offsets from global LEFT, TOP, FRONT                  */
        PMacc::SubGrid<DIM2> const & subGrid(
            PMacc::Environment<DIM2>::get().SubGrid());

        // Recall that the following is defined:
        //     typedef MappingDescription<DIM2, math::CT::Int<16,16>> MappingDesc;
        //
        // where math::CT::Int<16,16> is arbitrarily(!) chosen SuperCellSize and DIM2
        // is the dimension of the grid.
        // Expression of 2nd argument translates to DataSpace<DIM2>(16,16,0).
        // This is the guard size (here set to be one Supercell wide in all
        // directions). Meaning we have 16*16*(2*grid.x+2*grid.y+4) more
        // cells in GridLayout than in the SubGrid.
        PMacc::GridLayout<DIM2> layout(
            subGrid.getLocalDomain().size,
            MappingDesc::SuperCellSize::toRT());

        // getDataSpace will return DataSpace( grid.x +16+16, grid.y +16+16).
        // init stores the arguments internally in a MappingDesc private
        // variable which stores the layout regarding Core, Border and guard
        // in units of SuperCells to be used by the kernel to identify itself.
        evo.init(
            MappingDesc(layout.getDataSpace(), 1, 1));

        buff1.reset(new PMacc::GridBuffer<std::uint8_t, DIM2>(layout, false));
        buff2.reset(new PMacc::GridBuffer<std::uint8_t, DIM2>(layout, false));

        PMacc::DataSpace<DIM2> guardingCells(1, 1);
        for (uint32_t i(1); i < PMacc::traits::NumberOfExchanges<DIM2>::value; ++i)
        {
            // to check which number corresponds to which direction, you can
            // use the following member of class Mask like done in the two lines below:
            // DataSpace<DIM2>relVec = Mask::getRelativeDirections<DIM2>(i);
            // std::cout << "Direction:" << i << " => Vec: (" << relVec[0] << "," << relVec[1] << ")" << std::endl;
            // The result is: 1:right(1,0), 2:left(-1,0), 3:up(0,1), 4:up right(1,1), 5:(-1,1), 6:(0,-1), 7:(1,-1), 8:(-1,-1)

            buff1->addExchange(PMacc::GUARD, PMacc::Mask(i), guardingCells, BUFF1);
            buff2->addExchange(PMacc::GUARD, PMacc::Mask(i), guardingCells, BUFF2);
        }

        // Both next lines are defined in GatherSlice.hpp:
        // -gather saves the MessageHeader object
        // -Then do an Allgather for the gloabalRanks from GC, sort out
        // -inactive processes (second/boolean ,argument in gather.init) and
        // save new MPI_COMMUNICATOR created from these into private var.
        // -return if rank == 0
        MessageHeader header(gridSize, layout, subGrid.getLocalDomain().offset);
        isMaster = gather.init(header, true);

        /* Calls kernel to initialize random generator. Game of Life is then  *
         * initialized using uniform random numbers. With 10% (second arg)    *
         * white points. World will be written to buffer in first argument    */
        evo.initEvolution(buff1->getDeviceBuffer().getDataBox(), 0.1f);

    }

    void start()
    {
        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

        if(isMaster)
        {
            buff1->deviceToHost();

            // Write out the initial picture into a file.
            std::string sFileNameWithoutExt(
                "gol_" + std::to_string(gridSize.x())
                + "x" + std::to_string(gridSize.y())
                + "_" + alpaka::acc::getAccName<PMacc::AlpakaAcc<alpaka::dim::DimInt<2u>>>()
                + "_init");
            std::replace(sFileNameWithoutExt.begin(), sFileNameWithoutExt.end(), '<', '_');
            std::replace(sFileNameWithoutExt.begin(), sFileNameWithoutExt.end(), '>', '_');
            writeFullImage(*buff1.get(), sFileNameWithoutExt);
        }

        for(std::size_t i(0); i < steps; ++i)
        {
            oneStep(i, *buff1.get(), *buff2.get());
            std::swap(buff1, buff2);
        }
    }

private:
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    void oneStep(
        std::size_t const & currentStep,
        PMacc::GridBuffer<std::uint8_t, DIM2> & read,
        PMacc::GridBuffer<std::uint8_t, DIM2> & write)
    {
        std::cout << "step: " << currentStep << std::endl;

        auto splitEvent(__getTransactionEvent());
        // GridBuffer 'read' will use 'splitEvent' to schedule transaction
        // tasks from the Guard of this local Area to the Borders of the
        // neighboring areas added by 'addExchange'. All transactions in
        // Transaction Manager will then be done in parallel to the
        // calculations in the core. In order to synchronize the data
        // transfer for the case the core calculation is finished earlier,
        // GridBuffer.asyncComm returns a transaction handle we can check
        auto send(read.asyncCommunication(splitEvent));

        evo.run<PMacc::CORE>(
            read.getDeviceBuffer().getDataBox(),
            write.getDeviceBuffer().getDataBox());

        // Join communication with worker tasks, Now all following tasks run sequential
        __setTransactionEvent(send);

        // Calculate Borders
        evo.run<PMacc::BORDER>(
            read.getDeviceBuffer().getDataBox(),
            write.getDeviceBuffer().getDataBox());

        // Copy from device to host for saving. All threads and not only the master have to do this.
        write.deviceToHost();

        if(isMaster)
        {
            // Write out the picture into a file.
            std::string sFileNameWithoutExt1(
                "gol_" + std::to_string(write.getHostBuffer().getDataSpace().x())
                + "x" + std::to_string(write.getHostBuffer().getDataSpace().y())
                + "_" + alpaka::acc::getAccName<PMacc::AlpakaAcc<alpaka::dim::DimInt<2u>>>()
                + "_" + std::to_string(currentStep));
            GrayImgWriter imgWriter;
            std::replace(sFileNameWithoutExt1.begin(), sFileNameWithoutExt1.end(), '<', '_');
            std::replace(sFileNameWithoutExt1.begin(), sFileNameWithoutExt1.end(), '>', '_');
            imgWriter(write.getHostBuffer().getDataBox(), write.getHostBuffer().getDataSpace(), sFileNameWithoutExt1);

            // Write out the picture into a file.
            std::string sFileNameWithoutExt(
                "gol_" + std::to_string(gridSize.x())
                + "x" + std::to_string(gridSize.y())
                + "_" + alpaka::acc::getAccName<PMacc::AlpakaAcc<alpaka::dim::DimInt<2u>>>()
                + "_" + std::to_string(currentStep));
            std::replace(sFileNameWithoutExt.begin(), sFileNameWithoutExt.end(), '<', '_');
            std::replace(sFileNameWithoutExt.begin(), sFileNameWithoutExt.end(), '>', '_');
            writeFullImage(write, sFileNameWithoutExt);
        }
    }

    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    void writeFullImage(
        PMacc::GridBuffer<std::uint8_t, DIM2> & localGridBuffer,
        std::string const & sFileNameWithoutExt)
    {
        // gather::operator() gathers all the buffers and assembles those to a complete picture discarding the guards.
        auto picture(gather(localGridBuffer.getHostBuffer().getDataBox()));

        GrayImgWriter imgWriter;
        imgWriter(picture, gridSize, sFileNameWithoutExt);
    }

private:
    PMacc::DataSpace<DIM2> gridSize;
    Evolution<MappingDesc> evo;
    GatherSlice gather;

    // for storing black (dead) and white (alive) data for gol
    std::unique_ptr<PMacc::GridBuffer<std::uint8_t, DIM2>> buff1; // Buffer(\see types.h) for swapping between old and new world
    std::unique_ptr<PMacc::GridBuffer<std::uint8_t, DIM2>> buff2; // like evolve(buff2 &, const buff1) would work internally
    std::size_t const steps;

    bool isMaster;
};
}

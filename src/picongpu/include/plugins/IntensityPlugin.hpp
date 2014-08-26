/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"

#include "simulation_classTypes.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>

#include "plugins/ILightweightPlugin.hpp"

#include "fields/FieldE.hpp"
#include "memory/boxes/CachedBox.hpp"
#include "basicOperations.hpp"
#include "math/vector/compile-time/Int.hpp"
#include "dimensions/SuperCellDescription.hpp"
#include "math/vector/compile-time/Vector.hpp"

namespace picongpu
{
using namespace PMacc;

/* count particles in an area
 * is not optimized, it checks any partcile position if its realy a particle
 */
template<class FieldBox, class BoxMax, class BoxIntegral>
__global__ void kernelIntensity(FieldBox field, DataSpace<simDim> cellsCount, BoxMax boxMax, BoxIntegral integralBox)
{

    typedef MappingDesc::SuperCellSize SuperCellSize;
    __shared__ float_X s_integrated[SuperCellSize::y::value];
    __shared__ float_X s_max[SuperCellSize::y::value];

    __syncthreads(); /*wait that all shared memory is initialised*/

    /*descripe size of a worker block for cached memory*/
    typedef SuperCellDescription<
        PMacc::math::CT::Int<SuperCellSize::x::value,SuperCellSize::y::value>
        > SuperCell2D;

    PMACC_AUTO(s_field, CachedBox::create < 0, float> (SuperCell2D()));

    int y = blockIdx.y * SuperCellSize::y::value + threadIdx.y;
    int yGlobal = y + SuperCellSize::y::value;
    const DataSpace<DIM2> threadId(threadIdx);

    if (threadId.x() == 0)
    {
        /*clear destination arrays*/
        s_integrated[threadId.y()] = float_X(0.0);
        s_max[threadId.y()] = float_X(0.0);
    }
    __syncthreads();

    /*move cell wise over z direction(without garding cells)*/
    for (int z = GUARD_SIZE * SuperCellSize::z::value; z < cellsCount.z() - GUARD_SIZE * SuperCellSize::z::value; ++z)
    {
        /*move supercell wise over x direction without guarding*/
        for (int x = GUARD_SIZE * SuperCellSize::x::value + threadId.x(); x < cellsCount.x() - GUARD_SIZE * SuperCellSize::x::value; x += SuperCellSize::x::value)
        {
            const float3_X field_at_point(field(DataSpace<DIM3 > (x, yGlobal, z)));
            s_field(threadId) = math::abs2(field_at_point);
            __syncthreads();
            if (threadId.x() == 0)
            {
                /*master threads moves cell wise over 2D supercell*/
                for (int x_local = 0; x_local < SuperCellSize::x::value; ++x_local)
                {
                    DataSpace<DIM2> localId(x_local, threadId.y());
                    s_integrated[threadId.y()] += s_field(localId);
                    s_max[threadId.y()] = fmaxf(s_max[threadId.y()], s_field(localId));

                }
            }
        }
    }
    __syncthreads();

    if (threadId.x() == 0)
    {
        /*copy result to global array*/
        integralBox[y] = s_integrated[threadId.y()];
        boxMax[y] = s_max[threadId.y()];
    }


}

template<class FieldBox, class BoxMax, class BoxIntegral>
__host__ void wrapper_kernelIntensity(dim3 grid, dim3 block, FieldBox field, DataSpace<simDim> cellsCount, BoxMax boxMax, BoxIntegral integralBox)
{
        __cudaKernel(kernelIntensity<FieldBox>)(grid, block)(field, cellsCount, boxMax, integralBox);
}

class IntensityPlugin : public ILightweightPlugin
{
private:
    typedef MappingDesc::SuperCellSize SuperCellSize;


    GridBuffer<float, DIM1> *localMaxIntensity;
    GridBuffer<float, DIM1> *localIntegratedIntensity;
    MappingDesc *cellDescription;
    uint32_t notifyFrequency;

    std::string analyzerName;
    std::string analyzerPrefix;

    std::ofstream outFileMax;
    std::ofstream outFileIntegrated;
    /*only rank 0 create a file*/
    bool writeToFile;
public:

    /*! Calculate the max und integrated E-Field energy over laser propagation direction (in our case Y)
     * max is only the SI  value of the amplitude (V/m)
     * integrated is the integral of amplidude of X and Z on Y position (is V/m in cell volume)
     */
    IntensityPlugin(std::string name, std::string prefix) :
    analyzerName(name),
    analyzerPrefix(prefix),
    localMaxIntensity(NULL),
    localIntegratedIntensity(NULL),
    cellDescription(NULL),
    notifyFrequency(0),
    writeToFile(false)
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    virtual ~IntensityPlugin()
    {

    }

    void notify(uint32_t currentStep)
    {
        calcIntensity(currentStep);
        combineData(currentStep);
    }

    void pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ((analyzerPrefix + ".period").c_str(),
             po::value<uint32_t > (&notifyFrequency), "enable analyser [for each n-th step]");
    }

    std::string pluginGetName() const
    {
        return analyzerName;
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        this->cellDescription = cellDescription;
    }

private:

    void pluginLoad()
    {
        if (notifyFrequency > 0)
        {
            writeToFile = Environment<simDim>::get().GridController().getGlobalRank() == 0;
            int yCells = cellDescription->getGridLayout().getDataSpaceWithoutGuarding().y();

            localMaxIntensity = new GridBuffer<float, DIM1 > (DataSpace<DIM1 > (yCells)); //create one int on gpu und host
            localIntegratedIntensity = new GridBuffer<float, DIM1 > (DataSpace<DIM1 > (yCells)); //create one int on gpu und host

            if (writeToFile)
            {
                createFile(analyzerName + "_max.dat", outFileMax);
                createFile(analyzerName + "_integrated.dat", outFileIntegrated);
            }

            Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyFrequency);
        }
    }

    void pluginUnload()
    {
        if (notifyFrequency > 0)
        {
            if (writeToFile)
            {
                flushAndCloseFile(outFileIntegrated);
                flushAndCloseFile(outFileMax);
            }
            __delete(localMaxIntensity);
            __delete(localIntegratedIntensity);
        }
    }

private:

    /* reduce data from all gpus to one array
     * @param currentStep simulation step
     */
    void combineData(uint32_t currentStep)
    {

        const DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());
	Window window(MovingWindow::getInstance().getWindow( currentStep));

        PMACC_AUTO(simBox,Environment<simDim>::get().SubGrid().getSimulationBox());

        const int yGlobalSize = simBox.getGlobalSize().y();
        const int yLocalSize = localSize.y();

        const int gpus = Environment<simDim>::get().GridController().getGpuNodes().productOfComponents();


        /**\todo: fixme I cant work with not regular domains (use mpi_gatherv)*/
        DataSpace<simDim> globalRootCell(simBox.getGlobalOffset());
        int yOffset = globalRootCell.y();
        int* yOffsetsAll = new int[gpus];
        float* maxAll = new float[yGlobalSize];
        float* maxAllTmp = new float[yLocalSize * gpus];
        memset(maxAll, 0, sizeof (float) *yGlobalSize);
        float* integretedAll = new float[yGlobalSize];
        float* integretedAllTmp = new float[yLocalSize * gpus];
        memset(integretedAll, 0, sizeof (float) *yGlobalSize);

        MPI_CHECK(MPI_Gather(&yOffset, 1, MPI_INT, yOffsetsAll, 1,
                             MPI_INT, 0, MPI_COMM_WORLD));

        MPI_CHECK(MPI_Gather(localMaxIntensity->getHostBuffer().getBasePointer(), yLocalSize, MPI_FLOAT,
                             maxAllTmp, yLocalSize, MPI_FLOAT,
                             0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Gather(localIntegratedIntensity->getHostBuffer().getBasePointer(), yLocalSize, MPI_FLOAT,
                             integretedAllTmp, yLocalSize, MPI_FLOAT,
                             0, MPI_COMM_WORLD));

        if (writeToFile)
        {
            for (int i = 0; i < gpus; ++i)
            {
                int gOffset = yOffsetsAll[i];
                int tmpOff = yLocalSize*i;
                for (int y = 0; y < yLocalSize; ++y)
                {
                    maxAll[gOffset + y] = std::max(maxAllTmp[tmpOff + y], maxAll[gOffset + y]);
                    integretedAll[gOffset + y] += integretedAllTmp[tmpOff + y];
                }
            }

	    const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
	    size_t physicelYCellOffset = numSlides * yLocalSize + window.globalDimensions.offset.y();
            writeFile(currentStep,
                      maxAll + window.globalDimensions.offset.y(),
                      window.globalDimensions.size.y(),
                      physicelYCellOffset,
                      outFileMax,
                      UNIT_EFIELD
                      );

            double unit=UNIT_EFIELD*CELL_VOLUME*SI::EPS0_SI;
            for(uint32_t i=0;i<simDim;++i)
                unit*=UNIT_LENGTH;

            writeFile(currentStep,
                      integretedAll + window.globalDimensions.offset.y(),
                      window.globalDimensions.size.y(),
                      physicelYCellOffset,
                      outFileIntegrated,
                      unit
                      );
        }

        delete[] yOffsetsAll;
        delete[] maxAll;
        delete[] integretedAll;
        delete[] maxAllTmp;
        delete[] integretedAllTmp;
    }

    /* write data from array to a file
     * write current step to first column
     *
     * @param currentStep simulation step
     * @param array shifted source array (begin printing from first element)
     * @param count number of elements to print
     * @param physicalYOffset offset in cells to the absolute simulation begin
     * @param stream destination stream
     * @param unit unit to scale values from pic units to si units
     */
    void writeFile(size_t currentStep, float* array, size_t count, size_t physicalYOffset, std::ofstream& stream, double unit)
    {
        stream << currentStep << " ";
        for (size_t i = 0; i < count; ++i)
        {
            stream << (physicalYOffset + i) * SI::CELL_HEIGHT_SI << " ";
        }
        stream << std::endl << currentStep << " ";
        for (size_t i = 0; i < count; ++i)
        {
            stream << sqrt((double) (array[i])) * unit << " ";
        }
        stream << std::endl;
    }

    /* run calculation of intensity
     * sync all result data to host side
     *
     * @param currenstep simulation step
     */
    void calcIntensity(uint32_t)
    {
        DataConnector &dc = Environment<>::get().DataConnector();

        FieldE* fieldE = &(dc.getData<FieldE > (FieldE::getName(), true));

        /*start only worker for any supercell in laser propagation direction*/
        dim3 grid(1, cellDescription->getGridSuperCells().y() - cellDescription->getGuardingSuperCells());
        /*use only 2D slice XY for supercell handling*/
        typedef typename MappingDesc::SuperCellSize SuperCellSize;
        dim3 block(PMacc::math::CT::Vector<SuperCellSize::x,SuperCellSize::y>::toRT().toDim3());

	wrapper_kernelIntensity(grid, block, fieldE->getDeviceDataBox(),
             fieldE->getGridLayout().getDataSpace(),
             localMaxIntensity->getDeviceBuffer().getDataBox(),
             localIntegratedIntensity->getDeviceBuffer().getDataBox()
             );
/*
        __cudaKernel(kernelIntensity)
            (grid, block)
            (
             fieldE->getDeviceDataBox(),
             fieldE->getGridLayout().getDataSpace(),
             localMaxIntensity->getDeviceBuffer().getDataBox(),
             localIntegratedIntensity->getDeviceBuffer().getDataBox()
             );
*/
        localMaxIntensity->deviceToHost();
        localIntegratedIntensity->deviceToHost();

    }

    /*create a file with given filename
     * @param filename name of the output file
     * @param stream ref on a stream object
     */
    void createFile(std::string filename, std::ofstream& stream)
    {
        stream.open(filename.c_str(), std::ofstream::out | std::ostream::trunc);
        if (!stream)
        {
            std::cerr << "Can't open file [" << filename << "] for output, diasble analyser output. " << std::endl;
            writeToFile = false;
        }
        stream << "#step position_in_laser_propagation_direction" << std::endl;
        stream << "#step amplitude_data[*]" << std::endl;
    }

    /* close and flash a file stream object
     * @param stream stream which must closed
     */
    void flushAndCloseFile(std::ofstream& stream)
    {
        stream.flush();
        stream << std::endl; //now all data are written to file
        if (stream.fail())
            std::cerr << "Error on flushing file in IntensityPlugin. " << std::endl;
        stream.close();
    }

};

}


/**
 * Copyright 2015 Rene Widera
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

#include "simulation_defines.hpp"

#include "dimensions/DataSpace.hpp"

#include "mappings/kernel/AreaMapping.hpp"
#include "plugins/ILightweightPlugin.hpp"

#include "simulationControl/TimeInterval.hpp"

#include <cub/cub.cuh>

#include <string>

#define TIMING 0

namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;

template<class T_ParticleBox, class T_Mapping>
__global__ void kernelSortParticles(T_ParticleBox pb,
                                    T_Mapping mapper)
{

    typedef T_ParticleBox ParticleBox;
    typedef T_Mapping Mapping;
    typedef typename ParticleBox::FrameType FrameType;

    typedef typename MappingDesc::SuperCellSize Block;
    typedef typename Mapping::SuperCellSize SuperCellSize;
    const uint32_t cellsPerSuperCell = PMacc::math::CT::volume<SuperCellSize>::type::value;

    __shared__ FrameType* srcFrame;
    __shared__ FrameType* dstFrame;
    __shared__ bool isValid;

    typedef cub::BlockRadixSort<int, cellsPerSuperCell, 1, int> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    __shared__ int idx_storage[cellsPerSuperCell];

    __syncthreads(); /*wait that all shared memory is initialized*/


    const int linearThreadIdx = threadIdx.x;

    const DataSpace<simDim> superCellIdx(mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx)));
    if (linearThreadIdx == 0)
    {
        srcFrame = &(pb.getFirstFrame(superCellIdx, isValid));
        if (isValid)
        {
            dstFrame = &(pb.getEmptyFrame());
            pb.setAsFirstFrame(*dstFrame, superCellIdx);
        }
    }
    __syncthreads();
    if (!isValid)
        return; /* end kernel if we have no frames */

    while (isValid)
    {
        PMACC_AUTO(particleAtThreadIdx, (*srcFrame)[linearThreadIdx]);
        bool isParticles = particleAtThreadIdx[multiMask_];

        int localCellIdx[1];
        localCellIdx[0] = particleAtThreadIdx[localCellIdx_];
        int oldPlaceInFrame[1];
        oldPlaceInFrame[0] = linearThreadIdx;
        if (!isParticles)
        {
            localCellIdx[0] = cellsPerSuperCell; //mark as invalid
            oldPlaceInFrame[0] = -1;
        }


        BlockRadixSort(temp_storage).Sort(localCellIdx, oldPlaceInFrame);
        const int tmp = (linearThreadIdx * 32);
        idx_storage[tmp % cellsPerSuperCell + tmp / cellsPerSuperCell] = oldPlaceInFrame[0];
        //idx_storage[linearThreadIdx] = oldPlaceInFrame[0];
        __syncthreads();

        if (idx_storage[linearThreadIdx] != -1)
        {
            //PMACC_AUTO(srcParticle, (*srcFrame)[oldPlaceInFrame[0]]);
            PMACC_AUTO(srcParticle, (*srcFrame)[idx_storage[linearThreadIdx]]);
            PMACC_AUTO(dstParticle, (*dstFrame)[linearThreadIdx]);
            PMacc::particles::operations::assign(dstParticle, srcParticle);
            srcParticle[multiMask_] = 0;
        }

        __syncthreads();
        if (linearThreadIdx == 0)
        {
            dstFrame = srcFrame;
            srcFrame = &(pb.getNextFrame(*srcFrame, isValid));
        }
        __syncthreads();
    }
    if (linearThreadIdx == 0)
    {
        pb.removeLastFrame(superCellIdx);
    }

}

template<class ParticlesType>
class SortPartcilesPerFrame : public ILightweightPlugin
{
private:

    typedef MappingDesc::SuperCellSize SuperCellSize;

    ParticlesType *particles;

    GridBuffer<float_64, DIM1> *gBins;
    MappingDesc *cellDescription;

    std::string pluginName;
    std::string pluginPrefix;


    uint32_t notifyPeriod;


public:

    SortPartcilesPerFrame() :
    pluginName("SortPartcilesPerFrame: sort particle inside a frame"),
    pluginPrefix(ParticlesType::FrameType::getName() + std::string("_sortPerFrame")),
    particles(NULL),
    cellDescription(NULL),
    notifyPeriod(0)
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    virtual ~SortPartcilesPerFrame()
    {

    }

    void notify(uint32_t currentStep)
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        particles = &(dc.getData<ParticlesType > (ParticlesType::FrameType::getName(), true));

#if (TIMING == 1)
        __getTransactionEvent().waitForFinished();
        TimeIntervall timer;
#endif
        __picKernelArea(kernelSortParticles, *cellDescription, CORE + BORDER)
            (PMacc::math::CT::volume<SuperCellSize>::type::value)
            (particles->getDeviceParticlesBox());
        particles->template fillLastFrameGaps < CORE + BORDER > ();
#if (TIMING == 1)
        __getTransactionEvent().waitForFinished();
        timer.toggleEnd();
        std::cout << "step " << currentStep << "sort time: " <<
            timer.printInterval() << " = " <<
            (int) (timer.getInterval() / 1000.) << " sec" << std::endl;
#endif

    }

    void pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ((pluginPrefix + ".period").c_str(),
             po::value<uint32_t > (&notifyPeriod)->default_value(0),
             "enable plugin [for each n-th step]");
    }

    std::string pluginGetName() const
    {
        return pluginName;
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        this->cellDescription = cellDescription;
    }

private:

    void pluginLoad()
    {
        if (notifyPeriod > 0)
        {
            Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
        }
    }

    void pluginUnload()
    {

    }

};

} //namespace picongpu

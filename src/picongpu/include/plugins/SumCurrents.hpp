/**
 * Copyright 2013-2015 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Felix Schmitt, Benjamin Worpitz
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



#ifndef SUMCURRENTS_HPP
#define    SUMCURRENTS_HPP

#include <iostream>

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"

#include "simulation_classTypes.hpp"

#include "fields/FieldJ.hpp"

#include "dimensions/DataSpaceOperations.hpp"
#include "plugins/ILightweightPlugin.hpp"

namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;

typedef FieldJ::DataBoxType J_DataBox;

struct KernelSumCurrents
{
template<
    typename T_Acc,
    typename Mapping>
ALPAKA_FN_ACC void operator()(
    T_Acc const & acc,
    J_DataBox const & fieldJ,
    float3_X* gCurrent,
    Mapping const & mapper) const
{
    typedef typename Mapping::SuperCellSize SuperCellSize;

    DataSpace<simDim> const blockIndex(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc));
    DataSpace<simDim> const threadIndex(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));

    auto sh_sumJ(alpaka::block::shared::allocVar<float3_X>(acc));

    acc.syncBlockThreads(); /*wait that all shared memory is initialised*/

    const int linearThreadIdx = DataSpaceOperations<simDim>::template map<SuperCellSize > (threadIndex);

    if (linearThreadIdx == 0)
    {
        sh_sumJ = float3_X::create(0.0);
    }

    acc.syncBlockThreads();


    const DataSpace<simDim> superCellIdx(mapper.getSuperCellIndex(DataSpace<simDim > (blockIndex)));
    const DataSpace<simDim> cell(superCellIdx * SuperCellSize::toRT() + threadIndex);

    const float3_X myJ = fieldJ(cell);

    alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &(sh_sumJ.x()), myJ.x());
    alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &(sh_sumJ.y()), myJ.y());
    alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &(sh_sumJ.z()), myJ.z());

    acc.syncBlockThreads();

    if (linearThreadIdx == 0)
    {
        alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &(gCurrent->x()), sh_sumJ.x());
        alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &(gCurrent->y()), sh_sumJ.y());
        alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &(gCurrent->z()), sh_sumJ.z());
    }
}
};

class SumCurrents : public ILightweightPlugin
{
private:
    FieldJ* fieldJ;

    MappingDesc *cellDescription;
    uint32_t notifyFrequency;

    GridBuffer<float3_X, DIM1> *sumcurrents;

public:

    SumCurrents() :
    fieldJ(NULL),
    cellDescription(NULL),
    notifyFrequency(0)
    {

        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    virtual ~SumCurrents() = default;

    void notify(uint32_t currentStep)
    {
        DataConnector &dc = Environment<>::get().DataConnector();

        fieldJ = &(dc.getData<FieldJ > (FieldJ::getName(), true));


        const int rank = Environment<simDim>::get().GridController().getGlobalRank();
        const float3_X gCurrent = getSumCurrents();

        // gCurrent is just j
        // j = I/A
#if(SIMDIM==DIM3)
        const float3_X realCurrent(
                                   gCurrent.x() * CELL_HEIGHT * CELL_DEPTH,
                                   gCurrent.y() * CELL_WIDTH * CELL_DEPTH,
                                   gCurrent.z() * CELL_WIDTH * CELL_HEIGHT);
#elif(SIMDIM==DIM2)
        const float3_X realCurrent(
                                   gCurrent.x() * CELL_HEIGHT,
                                   gCurrent.y() * CELL_WIDTH,
                                   gCurrent.z() * CELL_WIDTH * CELL_HEIGHT);
#endif
        float3_64 realCurrent_SI(
                                 double(realCurrent.x()) * (UNIT_CHARGE / UNIT_TIME),
                                 double(realCurrent.y()) * (UNIT_CHARGE / UNIT_TIME),
                                 double(realCurrent.z()) * (UNIT_CHARGE / UNIT_TIME));

        /*FORMAT OUTPUT*/
        typedef std::numeric_limits< float_64 > dbl;

        std::cout.precision(dbl::digits10);
        if (math::abs(gCurrent.x()) + math::abs(gCurrent.y()) + math::abs(gCurrent.z()) != float_X(0.0))
            std::cout << "[ANALYSIS] [" << rank << "] [COUNTER] [SumCurrents] [" << currentStep
            << std::scientific << "] " <<
            realCurrent_SI << " Abs:" << math::abs(realCurrent_SI) << std::endl;
    }

    void pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ("sumcurr.period", po::value<uint32_t > (&notifyFrequency), "enable analyser [for each n-th step]");
    }

    std::string pluginGetName() const
    {
        return "SumCurrents";
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
            sumcurrents = new GridBuffer<float3_X, DIM1 > (DataSpace<DIM1 > (1)); //create one int on gpu und host

            Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyFrequency);
        }
    }

    void pluginUnload()
    {
        if (notifyFrequency > 0)
        {
            __delete(sumcurrents);
        }
    }

    float3_X getSumCurrents()
    {
        sumcurrents->getDeviceBuffer().setValue(float3_X::create(0.0));
        DataSpace<simDim> block(MappingDesc::SuperCellSize::toRT());

        KernelSumCurrents kernelSumCurrents;
        __picKernelArea(
            kernelSumCurrents,
            alpaka::dim::DimInt<simDim>,
            *cellDescription,
            CORE + BORDER,
            block)(
                fieldJ->getDeviceDataBox(),
                sumcurrents->getDeviceBuffer().getBasePointer());

        sumcurrents->deviceToHost();
        return sumcurrents->getHostBuffer().getDataBox()[0];
    }

};

}


#endif    /* SUMCURRENTS_HPP */


/**
 * Copyright 2013-2015 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Benjamin Worpitz
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

/**
 * @mainpage PIConGPU-Frame
 *
 * Project with HZDR for porting their PiC-code to a GPU cluster.
 *
 * \image html picongpu.jpg
 *
 * @author Heiko Burau, Rene Widera, Wolfgang Hoenig, Felix Schmitt, Axel Huebl, Michael Bussmann, Guido Juckeland, Benjamin Worpitz
 */

// includes heap configuration, all available policies, etc.
#include "mallocMC/mallocMC.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)

// configure the CreationPolicy "Scatter"
struct ScatterConfig
{
    /* 2MiB page can hold around 256 particle frames */
    typedef boost::mpl::int_<2*1024*1024> pagesize;
    /* accessblocks, regionsize and wastefactor are not finale selected
       and might be performance sensitive*/
    typedef boost::mpl::int_<4> accessblocks;
    typedef boost::mpl::int_<8> regionsize;
    typedef boost::mpl::int_<2> wastefactor;
    /* resetfreedpages is used to minimize memory fragmentation while different
       frame sizes were used*/
    typedef boost::mpl::bool_<true> resetfreedpages;
};

// Define a new allocator and call it ScatterAllocator
// which resembles the behavior of ScatterAlloc
using ScatterAllocator = mallocMC::Allocator<
    mallocMC::CreationPolicies::Scatter<ScatterConfig>,
    mallocMC::DistributionPolicies::Noop,
    mallocMC::OOMPolicies::ReturnNull,
    mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
    mallocMC::AlignmentPolicies::Shrink<>
    >;

//use ScatterAllocator to replace malloc/free
MALLOCMC_SET_ALLOCATOR_TYPE( ScatterAllocator );

#else

// Define a new allocator and call it ScatterAllocator
// which resembles the behavior of ScatterAlloc
using AllocatorHostNew = mallocMC::Allocator<
    mallocMC::CreationPolicies::HostNew,
    mallocMC::DistributionPolicies::Noop,
    mallocMC::OOMPolicies::ReturnNull,
    mallocMC::ReservePoolPolicies::NoOp,
    mallocMC::AlignmentPolicies::Noop
    >;

//use AllocatorHostNew to replace malloc/free
MALLOCMC_SET_ALLOCATOR_TYPE( AllocatorHostNew );

#endif

#include "communication/manager_common.h"
#include "ArgsParser.hpp"

#include <simulation_defines.hpp>
#include <mpi.h>


using namespace PMacc;
using namespace picongpu;

/*! start of PIConGPU
 *
 * @param argc count of arguments in argv
 * @param argv arguments of program start
 */
int main(int argc, char **argv)
{
    int prov;
    MPI_CHECK(MPI_Init_thread(&argc, &argv,MPI_THREAD_FUNNELED,&prov));
    std::cout<<"openmpi "<<prov<<"=="<<MPI_THREAD_FUNNELED<<std::endl;


    picongpu::simulation_starter::SimStarter sim;
    ArgsParser::ArgsErrorCode parserCode = sim.parseConfigs(argc, argv);
    int errorCode = 1;

    switch(parserCode)
    {
        case ArgsParser::ERROR:
            errorCode = 1;
            break;
        case ArgsParser::SUCCESS:
            sim.load();
            sim.start();
            sim.unload();
            /*set error code to valid (1) after the simulation terminates*/
        case ArgsParser::SUCCESS_EXIT:
            errorCode = 0;
            break;
    };

    MPI_CHECK(MPI_Finalize());
    return errorCode;
}

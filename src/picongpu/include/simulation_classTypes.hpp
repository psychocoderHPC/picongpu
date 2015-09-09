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

#pragma once

#include "debug/PIConGPUVerbose.hpp"
#include "simulation_defines.hpp"

#include "mappings/kernel/AreaMapping.hpp"
#include "math/Vector.hpp"
#include "eventSystem/EventSystem.hpp"
#include "types.h"

#include <boost/predef.h>

namespace picongpu
{
    using namespace PMacc;

    //short name for access verbose types of picongpu
    typedef PIConGPUVerbose picLog;

} //namespace picongpu

/**
 * Appends kernel arguments to the executor invocation and activates the kernel task.
 * If PMACC_SYNC_KERNEL is 1 cudaThreadSynchronize() is called before and after activation.
 *
 * @param ... Parameters to pass to kernel
 */
#if BOOST_COMP_MSVC
    #define PIC_KERNEL_PARAMS(...)\
            ,__VA_ARGS__, mapper));\
            PMACC_ACTIVATE_KERNEL();\
        }
#else
    #define PIC_KERNEL_PARAMS(...)\
            ,##__VA_ARGS__, mapper));\
            PMACC_ACTIVATE_KERNEL();\
        }
#endif

/**
 * Calls a CUDA kernel and creates an EventTask which represents the kernel.
 *
 * gridsize for kernel call is set by mapper
 * last argument of kernel call is add by mapper and is the mapper
 *
 * @param kernelname name of the CUDA kernel (can also used with templates etc. myKernnel<1>)
 * @param area area type for which the kernel is called
 */
#define __picKernelArea(KERNEL, DIM, description, area, block)\
    {\
        PMACC_KERNEL_CATCH(::alpaka::wait::wait(::PMacc::Environment<>::get().DeviceManager().getAccDevice()), "picKernelArea: crash before kernel call");\
        ::PMacc::AreaMapping<area, MappingDesc> mapper(description);\
        ::PMacc::TaskKernel * const taskKernel(::PMacc::Environment<>::get().Factory().createTaskKernel(#KERNEL));\
        auto const exec(::alpaka::exec::create<::PMacc::AlpakaAcc<DIM>>(::alpaka::workdiv::WorkDivMembers<DIM, AlpakaIdxSize>(mapper.getGridDim(), block), KERNEL\
        PIC_KERNEL_PARAMS

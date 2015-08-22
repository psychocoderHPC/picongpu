/**
 * Copyright 2013-2015 Felix Schmitt, Rene Widera, Benjamin Worpitz
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

#include "dimensions/DataSpace.hpp"
#include "eventSystem/EventSystem.hpp"
#include "ppFunctions.hpp"
#include "types.h"

#include <boost/predef.h>

/*
 * If this flag is defined all kernel calls would be checked and synchronize
 * this flag must set by the compiler or inside of the Makefile
 */
#if (PMACC_SYNC_KERNEL  == 1)
    #define PMACC_KERNEL_CATCH(COMMAND, MSG)\
        try\
        {\
            COMMAND;\
        }\
        catch(...)\
        {\
        }
#else
    /*no synchronize and check of kernel calls*/
    #define PMACC_KERNEL_CATCH(MSG, COMMAND)
#endif

/** Call activate kernel from taskKernel.
 *  If PMACC_SYNC_KERNEL is 1 cudaDeviceSynchronize() is called before
 *  and after activation.
 *
 * activateChecks is used if call is TaskKernel.waitforfinished();
 */
#define PMACC_ACTIVATE_KERNEL()\
    ::alpaka::stream::enqueue(taskKernel->getEventStream()->getCudaStream(), exec);\
    PMACC_KERNEL_CATCH(::alpaka::wait::wait(::PMacc::Environment<>::get().DeviceManager().getDevice()), "__cudaKernel: crash after kernel call");\
    taskKernel->activateChecks();\
    PMACC_KERNEL_CATCH(::alpaka::wait::wait(::PMacc::Environment<>::get().DeviceManager().getDevice()), "__cudaKernel: crash after kernel activation");\

/**
 * Appends kernel arguments to the executor invocation and activates the kernel task.
 * If PMACC_SYNC_KERNEL is 1 cudaThreadSynchronize() is called before and after activation.
 *
 * @param ... Parameters to pass to kernel
 */
#if BOOST_COMP_MSVC
    #define PMACC_KERNEL_PARAMS(...)\
            ,__VA_ARGS__));\
            PMACC_ACTIVATE_KERNEL();\
        }
#else
    #define PMACC_KERNEL_PARAMS(...)\
            ,##__VA_ARGS__));\
            PMACC_ACTIVATE_KERNEL();\
        }
#endif

/**
 * Calls a CUDA kernel and creates an EventTask which represents the kernel.
 *
 * @param KERNEL Instance of the kernel.
 */
#define __cudaKernel(KERNEL, DIM, ...)\
    {\
        PMACC_KERNEL_CATCH(::alpaka::wait::wait(::PMacc::Environment<>::get().DeviceManager().getDevice()), "__cudaKernel: crash before kernel call");\
        ::PMacc::TaskKernel * const taskKernel(::PMacc::Environment<>::get().Factory().createTaskKernel(#KERNEL));\
        auto const exec(::alpaka::exec::create<::PMacc::AlpakaAcc<DIM>>(::alpaka::workdiv::WorkDivMembers<DIM, ::PMacc::AlpakaSize>(__VA_ARGS__), KERNEL\
        PMACC_KERNEL_PARAMS

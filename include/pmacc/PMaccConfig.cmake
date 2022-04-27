# Copyright 2015-2022 Erik Zenker, Rene Widera, Axel Huebl, Jan Stephan
#
# This file is part of PMacc.
#
# PMacc is free software: you can redistribute it and/or modify
# it under the terms of either the GNU General Public License or
# the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PMacc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License and the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# and the GNU Lesser General Public License along with PMacc.
# If not, see <http://www.gnu.org/licenses/>.
#


# - Config file for the pmacc package
# It defines the following variables
#  PMacc_INCLUDE_DIRS - include directories for pmacc
#  PMacc_LIBRARIES    - libraries to link against
#  PMacc_DEFINITIONS  - definitions of pmacc

###############################################################################
# PMacc
###############################################################################
cmake_minimum_required(VERSION 3.15.0)

# set helper pathes to find libraries and packages
# Add specific hints
list(APPEND CMAKE_PREFIX_PATH "$ENV{MPI_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{VT_ROOT}")
# Add from environment after specific env vars
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")

# own modules for find_packages e.g. FindmallocMC
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PMacc_DIR}/../../thirdParty/cmake-modules/)


###############################################################################
# Build Flags
###############################################################################

set(PMACC_BUILD_TYPE "Release;Debug")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type for the project" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${PMACC_BUILD_TYPE}")
unset(PMACC_BUILD_TYPE)


################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
################################################################################

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()


###############################################################################
# Language Flags
###############################################################################

# enforce C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)


###############################################################################
# Definitely Unsupported Compilers
###############################################################################
# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message(FATAL_ERROR "GCC too old! Use GCC 4.9 or newer")
    endif()
# Clang
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
        message(FATAL_ERROR "Apple Clang (Xcode) too old! Use Xcode 8.0 or newer")
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.9)
        message(FATAL_ERROR "Clang too old! Use Clang 3.9 or newer")
    endif()
endif()


################################################################################
# alpaka path
################################################################################

# workaround for native CMake CUDA
# CMake is not forwarding CMAKE_CUDA_ARCHITECTURES to the CMake CUDA compiler check
# error: clang: error: cannot find libdevice for sm_20. Provide path to different CUDA installation via --cuda-path, or pass -nocudalib to build without linking with libdevice.
# The workaround is parsing CMAKE_CUDA_ARCHITECTURES and forward command line parameter directly to clang++.
if(alpaka_ACC_GPU_CUDA_ENABLE AND CMAKE_CUDA_COMPILER)
    string(REGEX MATCH "(.*clang.*)" IS_CLANGCUDA_COMPILER ${CMAKE_CUDA_COMPILER})
    if(IS_CLANGCUDA_COMPILER)
        foreach(_CUDA_ARCH_ELEM ${CMAKE_CUDA_ARCHITECTURES})
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cuda-gpu-arch=sm_${_CUDA_ARCH_ELEM}")
        endforeach()
    endif()
endif()

# workaround for a CMake bug which is not handled in alpaka 0.7.0
# https://github.com/alpaka-group/alpaka/pull/1423
if(alpaka_ACC_GPU_CUDA_ENABLE)
        include(CheckLanguage)
        check_language(CUDA)
        # Use user selected CMake CXX compiler as cuda host compiler to avoid fallback to the default system CXX host compiler.
        # CMAKE_CUDA_HOST_COMPILER is reset by check_language(CUDA) therefore definition passed by the user via -DCMAKE_CUDA_HOST_COMPILER are
        # ignored by CMake (looks like a CMake bug).
        # The if condition used here should work correct after the CMake bug is fixed, too.
        # Check the environment variable CUDAHOSTCXX to prefer the CUDA host compiler set by the user.
        if("$ENV{CUDAHOSTCXX}" STREQUAL "" AND NOT CMAKE_CUDA_HOST_COMPILER)
            set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
        endif()
        enable_language(CUDA)
endif()

# set path to internal
set(PMACC_alpaka_PROVIDER "intern" CACHE STRING "Select which alpaka is used")
set_property(CACHE PMACC_alpaka_PROVIDER PROPERTY STRINGS "intern;extern")
mark_as_advanced(PMACC_alpaka_PROVIDER)

if(${PMACC_alpaka_PROVIDER} STREQUAL "intern")
    list(INSERT CMAKE_MODULE_PATH 0 "${PMacc_DIR}/../../thirdParty/cupla/alpaka")
endif()

# Set alpaka CXX standard because the default is currently C++14.
if(NOT DEFINED alpaka_CXX_STANDARD)
    set(alpaka_CXX_STANDARD ${CMAKE_CXX_STANDARD} CACHE STRING "C++ standard version")
endif()

################################################################################
# Find cupla
################################################################################

# set path to internal
set(PMACC_CUPLA_PROVIDER "intern" CACHE STRING "Select which cupla is used")
set_property(CACHE PMACC_CUPLA_PROVIDER PROPERTY STRINGS "intern;extern")
mark_as_advanced(PMACC_CUPLA_PROVIDER)

# force activate CUDA backend if CMAKE_CUDA_ARCHITECTURES is defined
if(
    (CMAKE_CUDA_ARCHITECTURES) AND
    (NOT alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE) AND
    (NOT alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE) AND
    (NOT alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE) AND
    (NOT alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE) AND
    (NOT alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE) AND
    (NOT alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE) AND
    (NOT alpaka_ACC_CPU_BT_OMP4_ENABLE)
)
    option(alpaka_ACC_GPU_CUDA_ENABLE "Enable the CUDA GPU accelerator" ON)
    option(alpaka_ACC_GPU_CUDA_ONLY_MODE
        "Only back-ends using CUDA can be enabled in this mode \
        (This allows to mix alpaka code with native CUDA code)."
        ON)
endif()

if(${PMACC_CUPLA_PROVIDER} STREQUAL "intern")
    add_subdirectory(${PMacc_DIR}/../../thirdParty/cupla ${CMAKE_BINARY_DIR}/cupla)
else()
    find_package("cupla" PATHS $ENV{CUPLA_ROOT} REQUIRED)
endif()

# disable CUDA only mode if cuda backend is disabled
if((NOT alpaka_ACC_GPU_CUDA_ENABLE) AND alpaka_ACC_GPU_CUDA_ONLY_MODE)
    set(alpaka_ACC_GPU_CUDA_ONLY_MODE OFF CACHE BOOL
        "Only back-ends using CUDA can be enabled in this mode \
        (This allows to mix alpaka code with native CUDA code)."
        FORCE)
    message(WARNING "alpaka_ACC_GPU_CUDA_ONLY_MODE is set to OFF because cuda backend is not activated")
endif()

# add possible indirect/transient library dependencies from alpaka backends
# note: includes and definitions are already added in the cupla_add_executable
#       wrapper
set(PMacc_LIBRARIES ${PMacc_LIBRARIES} cupla::cupla)


###############################################################################
# CPU Architecture: available instruction sets for e.g. SIMD extensions
#
# Conveniently set the architecture for the CPU compiler via this option.
# For unsupported compilers, ignore this option and set CXXFLAGS.
###############################################################################

set(PMACC_CPU_ARCH $ENV{PMACC_CPU_ARCH} CACHE STRING
    "compiler dependent CPU architecture string"
)

# list of known compiler flags to set the CPU architecture
# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "ppc64le")
        set(PMACC_CPU_ARCH_TEMPLATE "-mcpu={} -mtune={}")
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
        set(PMACC_CPU_ARCH_TEMPLATE "-mcpu={}")
    else()
        set(PMACC_CPU_ARCH_TEMPLATE "-march={} -mtune={}")
    endif()
# ICC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    if("${PMACC_CPU_ARCH}" STREQUAL "native")
        set(PMACC_CPU_ARCH_TEMPLATE "-march={} -mtune={}")
    else()
        set(PMACC_CPU_ARCH_TEMPLATE "-x{}")
    endif()
# Clang
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
        set(PMACC_CPU_ARCH_TEMPLATE "-mcpu={}")
    else()
        set(PMACC_CPU_ARCH_TEMPLATE "-march={} -mtune={}")
    endif()
# XL
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "XL")
    set(PMACC_CPU_ARCH_TEMPLATE "-qarch={}")
# PGI
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    if(NOT "${PMACC_CPU_ARCH}" STREQUAL "native")
        set(PMACC_CPU_ARCH_TEMPLATE "-tp={}")
    endif()
endif()

# architecture is set and compiler is known
if(PMACC_CPU_ARCH AND PMACC_CPU_ARCH_TEMPLATE)
    string(REPLACE
       "{}"
       "${PMACC_CPU_ARCH}"
       PMACC_CPU_ARCH_STRING
       "${PMACC_CPU_ARCH_TEMPLATE}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PMACC_CPU_ARCH_STRING}")
endif()


################################################################################
# VampirTrace
################################################################################

option(VAMPIR_ENABLE "Create PMacc with VampirTrace support" OFF)

# set filters: please do NOT use line breaks WITHIN the string!
set(VT_INST_FILE_FILTER
    "stl,usr/include,libgpugrid,vector_types.h,Vector.hpp,DeviceBuffer.hpp,DeviceBufferIntern.hpp,Buffer.hpp,StrideMapping.hpp,StrideMappingMethods.hpp,MappingDescription.hpp,AreaMapping.hpp,AreaMappingMethods.hpp,ExchangeMapping.hpp,ExchangeMappingMethods.hpp,DataSpace.hpp,Manager.hpp,Manager.tpp,Transaction.hpp,Transaction.tpp,TransactionManager.hpp,TransactionManager.tpp,Vector.tpp,Mask.hpp,ITask.hpp,EventTask.hpp,EventTask.tpp,StandardAccessor.hpp,StandardNavigator.hpp,HostBuffer.hpp,HostBufferIntern.hpp"
    CACHE STRING "VampirTrace: Files to exclude from instrumentation")
set(VT_INST_FUNC_FILTER
    "vector,Vector,dim3,GPUGrid,execute,allocator,Task,Manager,Transaction,Mask,operator,DataSpace,PitchedBox,Event,new,getGridDim,GetCurrentDataSpaces,MappingDescription,getOffset,getParticlesBuffer,getDataSpace,getInstance"
    CACHE STRING "VampirTrace: Functions to exclude from instrumentation")

if(VAMPIR_ENABLE)
    message(STATUS "Building with VampirTrace support")
    set(VAMPIR_ROOT "$ENV{VT_ROOT}")
    if(NOT VAMPIR_ROOT)
        message(FATAL_ERROR "Environment variable VT_ROOT not set!")
    endif(NOT VAMPIR_ROOT)

    # compile flags
    execute_process(COMMAND $ENV{VT_ROOT}/bin/vtc++ -vt:hyb -vt:showme-compile
                    OUTPUT_VARIABLE VT_COMPILEFLAGS
                    RESULT_VARIABLE VT_CONFIG_RETURN
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT VT_CONFIG_RETURN EQUAL 0)
        message(FATAL_ERROR "Can NOT execute 'vtc++' at $ENV{VT_ROOT}/bin/vtc++ - check file permissions")
    endif()
    # link flags
    execute_process(COMMAND $ENV{VT_ROOT}/bin/vtc++ -vt:hyb -vt:showme-link
                    OUTPUT_VARIABLE VT_LINKFLAGS
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    # bugfix showme
    string(REPLACE "--as-needed" "--no-as-needed" VT_LINKFLAGS "${VT_LINKFLAGS}")

    # modify our flags
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${VT_LINKFLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VT_COMPILEFLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finstrument-functions-exclude-file-list=${VT_INST_FILE_FILTER}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finstrument-functions-exclude-function-list=${VT_INST_FUNC_FILTER}")

    # nvcc flags (rly necessary?)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -Xcompiler=-finstrument-functions,-finstrument-functions-exclude-file-list=\\\"${VT_INST_FILE_FILTER}\\\"
        -Xcompiler=-finstrument-functions-exclude-function-list=\\\"${VT_INST_FUNC_FILTER}\\\"
        -Xcompiler=-DVTRACE -Xcompiler=-I\\\"${VT_ROOT}/include/vampirtrace\\\"
        -v)

    # for manual instrumentation and hints that vampir is enabled in our code
    set(PMacc_DEFINITIONS ${PMacc_DEFINITIONS} -DVTRACE)
endif(VAMPIR_ENABLE)


################################################################################
# Find MPI
################################################################################

find_package(MPI REQUIRED)
set(PMacc_INCLUDE_DIRS ${PMacc_INCLUDE_DIRS} ${MPI_C_INCLUDE_PATH})
set(PMacc_LIBRARIES ${PMacc_LIBRARIES} ${MPI_C_LIBRARIES})

# bullxmpi fails if it can not find its c++ counter part
if(MPI_CXX_FOUND)
    set(PMacc_LIBRARIES ${PMacc_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif(MPI_CXX_FOUND)


################################################################################
# Find Boost
################################################################################

find_package(Boost 1.66 REQUIRED COMPONENTS filesystem system math_tr1)
if(TARGET Boost::filesystem)
    set(PMacc_LIBRARIES ${PMacc_LIBRARIES} Boost::boost Boost::filesystem
                                           Boost::system Boost::math_tr1)
else()
    set(PMacc_INCLUDE_DIRS ${PMacc_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
    set(PMacc_LIBRARIES ${PMacc_LIBRARIES} ${Boost_LIBRARIES})
endif()

# Boost 1.55 added support for a define that makes result_of look for
# the result<> template and falls back to decltype if none is found. This is
# great for the transition from the "wrong" usage to the "correct" one as
message(STATUS "Boost: result_of with TR1 style and decltype fallback")
set(PMacc_DEFINITIONS ${PMacc_DEFINITIONS} -DBOOST_RESULT_OF_USE_TR1_WITH_DECLTYPE_FALLBACK)

# We do not use std::auto_ptr and keeping this enabled in Boost causes a
# warning with NVCC+GCC and is unnecessary time spend in compile time
# (note that std::auto_ptr is deprecated in C++11 and removed in C++17)
message(STATUS "Boost: deactivate std::auto_ptr")
set(PMacc_DEFINITIONS ${PMacc_DEFINITIONS} -DBOOST_NO_AUTO_PTR)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message(STATUS "Boost: Disable variadic templates")
    message(STATUS "Boost: Do not use fenv.h from standard library")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_VARIADIC_TEMPLATES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_VARIADIC_TEMPLATES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_FENV_H")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # suppress boost error
    # 'no member named "impl" in "boost::detail::thread_move_t<boost::detail::nullary_function<void ()> >"'
    # in 'boost/thread/detail/nullary_function.hpp'
    message(STATUS "Boost: Do not use C++11 smart pointers from standard library")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_SMART_PTR")
endif()

# Newer Boost releases: probably troublesome, warn at least
if(Boost_VERSION GREATER 107000)
    message(WARNING "Untested Boost release > 1.70.0 (Found ${Boost_VERSION})! "
                    "Maybe use a newer PIConGPU?")
endif()

################################################################################
# Find OpenMP
################################################################################

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND (alpaka_ACC_GPU_HIP_ENABLE OR (alpaka_ACC_GPU_CUDA_ENABLE AND alpaka_CUDA_COMPILER MATCHES "clang")))
    # For HIP the problem is that in alpaka '::isnan(), ::sinh(), ::isfinite(), ::isinf()' is not found.
    # The reason could be that if OpenMP is activated clang is using math C headers where all of these functions are macros.
    message(WARNING "OpenMP host side acceleration is disabled: CUDA/HIP compilation with clang is not supporting OpenMP.")
else()
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()


################################################################################
# Find mallocMC
################################################################################

if(alpaka_ACC_GPU_CUDA_ENABLE OR alpaka_ACC_GPU_HIP_ENABLE)
    set(mallocMC_alpaka_PROVIDER "extern" CACHE STRING "Select which alpaka is used for mallocMC")
    find_package(mallocMC 2.5.0 QUIET)

    if(NOT mallocMC_FOUND)
        message(STATUS "Using mallocMC from thirdParty/ directory")
        set(MALLOCMC_ROOT "${PMacc_DIR}/../../thirdParty/mallocMC")
        find_package(mallocMC 2.5.0 REQUIRED)
    endif(NOT mallocMC_FOUND)

    set(PMacc_INCLUDE_DIRS ${PMacc_INCLUDE_DIRS} ${mallocMC_INCLUDE_DIRS})
    set(PMacc_LIBRARIES ${PMacc_LIBRARIES} ${mallocMC_LIBRARIES})
    set(PMacc_DEFINITIONS ${PMacc_DEFINITIONS} ${mallocMC_DEFINITIONS})
endif()


################################################################################
# PMacc options
################################################################################

option(PMACC_BLOCKING_KERNEL
    "activate checks for every kernel call and synch after every kernel call" OFF)
if(PMACC_BLOCKING_KERNEL)
    set(PMacc_DEFINITIONS ${PMacc_DEFINITIONS} "-DPMACC_SYNC_KERNEL=1")
endif(PMACC_BLOCKING_KERNEL)

set(PMACC_VERBOSE "0" CACHE STRING "set verbose level for PMacc")
set(PMacc_DEFINITIONS ${PMacc_DEFINITIONS} "-DPMACC_VERBOSE_LVL=${PMACC_VERBOSE}")

# PMacc header files
set(PMacc_INCLUDE_DIRS ${PMacc_INCLUDE_DIRS} "${PMacc_DIR}/..")

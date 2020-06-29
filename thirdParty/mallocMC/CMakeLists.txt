project(mallocMC)
cmake_minimum_required(VERSION 3.11.4)

# helper for libs and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/"
    "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")


################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
################################################################################

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

# Add module search path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/modules/")


set(mallocMC_CXX_STANDARD "11" CACHE STRING "C++ standard version")
set_property(CACHE mallocMC_CXX_STANDARD PROPERTY STRINGS "11;14;17;20")

###############################################################################
# HIP
###############################################################################

# Drop-down combo box in cmake-gui for HIP platforms.
set(mallocMC_HIP_PLATFORM "clang" CACHE STRING "Specify HIP platform")
SET_PROPERTY(CACHE mallocMC_HIP_PLATFORM PROPERTY STRINGS "nvcc;hcc;clang")

set(HIP_PLATFORM "${mallocMC_HIP_PLATFORM}" CACHE STRING "")

find_package(HIP 1.5 REQUIRED)
if(HIP_FOUND)
    set(HIP_HIPCC_FLAGS)

    if(mallocMC_HIP_PLATFORM MATCHES "nvcc")
        find_package(CUDA)
        if(NOT CUDA_FOUND)
            message(WARNING "Could not found CUDA while HIP platform is set to nvcc. Compiling might fail.")
        endif()

        set(mallocMC_CUDA_ARCH "30" CACHE STRING "GPU architecture")

        if(CUDA_VERSION VERSION_LESS 9.0)
            message(WARNING "CUDA Toolkit < 9.0 is not supported!")
        endif()

        list(APPEND HIP_NVCC_FLAGS "--expt-extended-lambda")
        list(APPEND HIP_NVCC_FLAGS "--expt-relaxed-constexpr")

        FOREACH(_HIP_ARCH_ELEM ${mallocMC_CUDA_ARCH})
            # set flags to create device code for the given architecture
            list(APPEND CUDA_NVCC_FLAGS
                --generate-code arch=compute_${_HIP_ARCH_ELEM},code=sm_${_HIP_ARCH_ELEM}
                --generate-code arch=compute_${_HIP_ARCH_ELEM},code=compute_${_HIP_ARCH_ELEM}
            )
        ENDFOREACH()
        # for CUDA cmake adds automatically compiler flags as nvcc does not do this,
        # but for HIP we have to do this here
        list(APPEND HIP_NVCC_FLAGS "-D__CUDACC__")
        list(APPEND HIP_NVCC_FLAGS "-ccbin ${CMAKE_CXX_COMPILER}")
        list(APPEND HIP_NVCC_FLAGS "-Xcompiler" "-g")

        if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            list(APPEND HIP_HIPCC_FLAGS "-G")
        endif()
        # propage host flags
        # set(CUDA_PROPAGATE_HOST_FLAGS ON) # does not exist in HIP, so do it manually
        string(TOUPPER "${CMAKE_BUILD_TYPE}" build_config)
        foreach( _flag ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${build_config}})
            list(APPEND HIP_NVCC_FLAGS "-Xcompiler ${_flag}")
        endforeach()

        if(CUDA_VERSION GREATER_EQUAL 9.0)
            # avoids warnings on host-device signatured, default constructors/destructors
            list(APPEND HIP_HIPCC_FLAGS "-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
        endif()

    endif() # nvcc

    list(APPEND HIP_HIPCC_FLAGS "-D__HIPCC__")
    list(APPEND HIP_HIPCC_FLAGS "-std=c++${mallocMC_CXX_STANDARD}")

    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        list(APPEND HIP_HIPCC_FLAGS "-g")
    endif()

    HIP_INCLUDE_DIRECTORIES(
        ${HIP_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
    )

    if(mallocMC_HIP_PLATFORM MATCHES "hcc" OR mallocMC_HIP_PLATFORM MATCHES "clang")
        # GFX600, GFX601, GFX700, GFX701, GFX702, GFX703, GFX704, GFX801, GFX802, GFX803, GFX810, GFX900, GFX902
        set(LIBS "${LIBS}" "--amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906")
    endif()
endif()

###############################################################################
# Boost
###############################################################################
find_package(Boost 1.48.0 REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

# nvcc + boost 1.55 work around
if(Boost_VERSION EQUAL 105500)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \"-DBOOST_NOINLINE=__attribute__((noinline))\" ")
endif(Boost_VERSION EQUAL 105500)


################################################################################
# Warnings
################################################################################
# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
  # new warning in gcc 4.8 (flag ignored in previous version)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
# ICC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_VARIADIC_TEMPLATES")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_VARIADIC_TEMPLATES")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_FENV_H")
# PGI
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minform=inform")
endif()


###############################################################################
# Installation
###############################################################################

# copy include folder to install folder
INSTALL(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/include/."
  DESTINATION include
  PATTERN ".git" EXCLUDE
  PATTERN "mallocMC_config.hpp" EXCLUDE
  )


###############################################################################
# Executables
###############################################################################
add_custom_target(examples DEPENDS mallocMC_Example01 mallocMC_Example02 mallocMC_Example03)

SET_SOURCE_FILES_PROPERTIES(examples/mallocMC_example01.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT OBJ)
SET_SOURCE_FILES_PROPERTIES(examples/mallocMC_example02.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT OBJ)
SET_SOURCE_FILES_PROPERTIES(examples/mallocMC_example03.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT OBJ)

HIP_ADD_EXECUTABLE(mallocMC_Example01
                    EXCLUDE_FROM_ALL
                    examples/mallocMC_example01.cpp )
HIP_ADD_EXECUTABLE(mallocMC_Example02
                    EXCLUDE_FROM_ALL
                    examples/mallocMC_example02.cpp )
HIP_ADD_EXECUTABLE(mallocMC_Example03
                    EXCLUDE_FROM_ALL
                    examples/mallocMC_example03.cpp )
#HIP_ADD_EXECUTABLE(VerifyHeap
#                    EXCLUDE_FROM_ALL
#                    tests/verify_heap.cpp )

target_link_libraries(mallocMC_Example01 ${LIBS})
target_link_libraries(mallocMC_Example02 ${LIBS})
target_link_libraries(mallocMC_Example03 ${LIBS})
#target_link_libraries(VerifyHeap ${LIBS})

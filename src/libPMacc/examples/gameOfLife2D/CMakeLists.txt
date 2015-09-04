#
# Copyright 2013-2015 Rene Widera, Benjamin Worpitz
#
# This file is part of libPMacc.
#
# libPMacc is free software: you can redistribute it and/or modify
# it under the terms of either the GNU General Public License or
# the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# libPMacc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License and the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# and the GNU Lesser General Public License along with libPMacc.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################
# Required CMake version
################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12.2)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project
################################################################################

PROJECT(gameOfLife)

# Set helper paths to find libraries and packages.
LIST(APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu" "$ENV{MPI_ROOT}" "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../../thirdParty/cmake-modules")

################################################################################
# Configure Dependencies
################################################################################

#-------------------------------------------------------------------------------
# VampirTrace
#-------------------------------------------------------------------------------
OPTION(VAMPIR_ENABLE "Create PIConGPU with VampirTrace support" OFF)

# SET filters: please do NOT use line breaks WITHIN the string!
SET(VT_INST_FILE_FILTER
    "stl,usr/include,libgpugrid,vector_types.h,Vector.hpp,DeviceBufferIntern.hpp,DeviceBuffer.hpp,Buffer.hpp,StrideMapping.hpp,StrideMappingMethods.hpp,MappingDescription.hpp,AreaMapping.hpp,AreaMappingMethods.hpp,ExchangeMapping.hpp,ExchangeMappingMethods.hpp,DataSpace.hpp,Manager.hpp,Manager.tpp,Transaction.hpp,Transaction.tpp,TransactionManager.hpp,TransactionManager.tpp,Vector.tpp,Mask.hpp,ITask.hpp,EventTask.hpp,EventTask.tpp,StandardAccessor.hpp,StandardNavigator.hpp,HostBufferIntern.hpp,HostBuffer.hpp"
    CACHE STRING "VampirTrace: Files to exclude from instrumentation")
SET(VT_INST_FUNC_FILTER
    "vector,Vector,dim3,GPUGrid,execute,allocator,Task,Manager,Transaction,Mask,operator,DataSpace,PitchedBox,CudaEvent,new,getGridDim,GetCurrentDataSpaces,MappingDescription,getOffset,getParticlesBuffer,getDataSpace,getInstance"
    CACHE STRING "VampirTrace: Functions to exclude from instrumentation")

IF(VAMPIR_ENABLE)
    MESSAGE(STATUS "Building with VampirTrace support")
    SET(VAMPIR_ROOT "$ENV{VT_ROOT}")
    IF(NOT VAMPIR_ROOT)
        MESSAGE(FATAL_ERROR "Environment variable VT_ROOT not SET!")
    ENDIF()

    # compile flags
    EXECUTE_PROCESS(COMMAND $ENV{VT_ROOT}/bin/vtc++ -vt:hyb -vt:showme-compile
                    OUTPUT_VARIABLE VT_COMPILEFLAGS
                    RESULT_VARIABLE VT_CONFIG_RETURN
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    IF(NOT VT_CONFIG_RETURN EQUAL 0)
        MESSAGE(FATAL_ERROR "Can NOT execute 'vtc++' at $ENV{VT_ROOT}/bin/vtc++ - check FILE permissions")
    ENDIF()
    # link flags
    EXECUTE_PROCESS(COMMAND $ENV{VT_ROOT}/bin/vtc++ -vt:hyb -vt:showme-link
                    OUTPUT_VARIABLE VT_LINKFLAGS
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    # bugfix showme
    string(REPLACE "--as-needed" "--no-as-needed" VT_LINKFLAGS "${VT_LINKFLAGS}")

    # modify our flags
    LIST(APPEND CMAKE_CXX_LINK_FLAGS ${VT_LINKFLAGS})
    LIST(APPEND CMAKE_CXX_FLAGS ${VT_COMPILEFLAGS})
    LIST(APPEND CMAKE_CXX_FLAGS "-finstrument-functions-exclude-FILE-list=${VT_INST_FILE_FILTER}")
    LIST(APPEND CMAKE_CXX_FLAGS "-finstrument-functions-exclude-function-list=${VT_INST_FUNC_FILTER}")

    # nvcc flags (rly necessary?)
    LIST(APPEND CUDA_NVCC_FLAGS
        -Xcompiler=-finstrument-functions,-finstrument-functions-exclude-FILE-list=\\\"${VT_INST_FILE_FILTER}\\\"
        -Xcompiler=-finstrument-functions-exclude-function-list=\\\"${VT_INST_FUNC_FILTER}\\\"
        -Xcompiler=-DVTRACE -Xcompiler=-I\\\"${VT_ROOT}/include/vampirtrace\\\"
        -v)

    # for manual instrumentation and hints that vampir is enabled in our code
    LIST(APPEND _GOL_DEFINITIONS_PRIVATE VTRACE)

    # titan work around: currently (5.14.4) the -D defines are not provided by -vt:showme-compile
    LIST(APPEND _GOL_DEFINITIONS_PRIVATE MPICH_IGNORE_CXX_SEEK)
ENDIF(VAMPIR_ENABLE)

#-------------------------------------------------------------------------------
# Score-P
#-------------------------------------------------------------------------------
OPTION(SCOREP_ENABLE "Create PIConGPU with Score-P support" OFF)

IF(SCOREP_ENABLE)
    MESSAGE(STATUS "Building with Score-P support")
    SET(SCOREP_ROOT "$ENV{SCOREP_ROOT}")
    IF(NOT SCOREP_ROOT)
        MESSAGE(FATAL_ERROR "Environment variable SCOREP_ROOT not SET!")
    ENDIF()

    # compile flags
    EXECUTE_PROCESS(COMMAND $ENV{SCOREP_ROOT}/bin/scorep-config --nocompiler --cflags
                    OUTPUT_VARIABLE SCOREP_COMPILEFLAGS
                    RESULT_VARIABLE SCOREP_CONFIG_RETURN
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    IF(NOT SCOREP_CONFIG_RETURN EQUAL 0)
        MESSAGE(FATAL_ERROR "Can NOT execute 'scorep-config' at $ENV{SCOREP_ROOT}/bin/scorep-config - check FILE permissions")
    ENDIF()

    # TODO: --cuda has to be added conditionally when it is enabled in alpaka.
    # link flags
    EXECUTE_PROCESS(COMMAND $ENV{SCOREP_ROOT}/bin/scorep-config --cuda --mpp=mpi --ldflags
                    OUTPUT_VARIABLE SCOREP_LINKFLAGS
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    # libraries
    EXECUTE_PROCESS(COMMAND $ENV{SCOREP_ROOT}/bin/scorep-config --cuda --mpp=mpi --libs
                    OUTPUT_VARIABLE SCOREP_LIBFLAGS
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(STRIP "${SCOREP_LIBFLAGS}" SCOREP_LIBFLAGS)

    # subsystem initialization FILE
    EXECUTE_PROCESS(COMMAND $ENV{SCOREP_ROOT}/bin/scorep-config --cuda --mpp=mpi --adapter-init
                    OUTPUT_VARIABLE SCOREP_INIT_FILE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    FILE(WRITE "${CMAKE_BINARY_DIR}/scorep_init.c" "${SCOREP_INIT_FILE}")

    # modify our flags
    LIST(APPEND CMAKE_CXX_LINK_FLAGS ${SCOREP_LINKFLAGS})
    LIST(APPEND CMAKE_CXX_FLAGS {SCOREP_COMPILEFLAGS})
    
    LIST(APPEND _GOL_FILES_SOURCE "${CMAKE_BINARY_DIR}/scorep_init.c")
ENDIF()

#-------------------------------------------------------------------------------
# Find Boost.
#-------------------------------------------------------------------------------
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options regex system filesystem)
LIST(APPEND _GOL_INCLUDE_DIRECTORIES_PRIVATE ${Boost_INCLUDE_DIRS})
LIST(APPEND _GOL_LIBRARIES_PRIVATE ${Boost_LIBRARIES})

#-------------------------------------------------------------------------------
# Find PMacc.
#-------------------------------------------------------------------------------
SET("PMACC_ROOT" "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE STRING  "The location of the PMacc library")

FIND_PACKAGE(PMacc REQUIRED)
LIST(APPEND _GOL_DEFINITIONS_PRIVATE ${PMacc_DEFINITIONS})
LIST(APPEND _GOL_INCLUDE_DIRECTORIES_PRIVATE ${PMacc_INCLUDE_DIRS})
LIST(APPEND _GOL_LIBRARIES_PRIVATE ${PMacc_LIBRARIES})

#-------------------------------------------------------------------------------
# Find PNGwriter
#-------------------------------------------------------------------------------
FIND_PACKAGE(PNGwriter)
IF(PNGwriter_FOUND)
    LIST(APPEND _GOL_DEFINITIONS_PRIVATE ${PNGwriter_DEFINITIONS} "GOL_ENABLE_PNG=1")
    LIST(APPEND _GOL_INCLUDE_DIRECTORIES_PRIVATE ${PNGwriter_INCLUDE_DIRS})
    LIST(APPEND _GOL_LIBRARIES_PRIVATE ${PNGwriter_LIBRARIES})
ENDIF(PNGwriter_FOUND)

################################################################################
# Compile and link.
################################################################################
# Add all the include files in all recursive subdirectories and group them accordingly for MSVC projects.
append_recursive_files_add_to_src_group("${CMAKE_CURRENT_LIST_DIR}/include" "${CMAKE_CURRENT_LIST_DIR}" "hpp" _GOL_FILES_HEADER)
append_recursive_files_add_to_src_group("${CMAKE_CURRENT_LIST_DIR}/src" "${CMAKE_CURRENT_LIST_DIR}" "cpp" _GOL_FILES_SOURCE)
append_recursive_files_add_to_src_group("${CMAKE_CURRENT_LIST_DIR}/src" "${CMAKE_CURRENT_LIST_DIR}" "cu" _GOL_FILES_SOURCE)

LIST(APPEND _GOL_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")
LIST(APPEND _GOL_INCLUDE_DIRECTORIES_PRIVATE ${_GOL_INCLUDE_DIR})

ADD_DEFINITIONS(
    ${_GOL_DEFINITIONS_PRIVATE})
INCLUDE_DIRECTORIES(
    ${_GOL_INCLUDE_DIRECTORIES_PRIVATE})

IF(ALPAKA_ACC_GPU_CUDA_ENABLE)
    # Force the main.cpp file to be recognized as header and not be compiled so there wont be a second entry point.
    SET_SOURCE_FILES_PROPERTIES(
        "${CMAKE_CURRENT_LIST_DIR}/src/main.cpp"
        PROPERTIES HEADER_FILE_ONLY TRUE)
    CMAKE_POLICY(SET CMP0023 OLD)   # CUDA_ADD_EXECUTABLE calls TARGET_LINK_LIBRARIES without keywords.
    CUDA_ADD_EXECUTABLE(
        "gameOfLife"
        ${_GOL_FILES_HEADER}
        ${_GOL_FILES_SOURCE})
ELSE()
    # Force the main.cu file to be recognized as header and not be compiled so there wont be a second entry point.
    SET_SOURCE_FILES_PROPERTIES(
        "${CMAKE_CURRENT_LIST_DIR}/src/main.cu"
        PROPERTIES HEADER_FILE_ONLY TRUE)
    ADD_EXECUTABLE(
        "gameOfLife"
        ${_GOL_FILES_HEADER}
        ${_GOL_FILES_SOURCE})
ENDIF()

TARGET_LINK_LIBRARIES(
    "gameOfLife"
    ${_GOL_LIBRARIES_PRIVATE})

################################################################################
# Install
################################################################################
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
ENDIF()

SET(GOL_COPY_ON_INSTALL "submit" CACHE LIST "folder which copied on install to install path" )

INSTALL(
    TARGETS gameOfLife
    RUNTIME DESTINATION bin)

#copy all subfolders (defined in: GOL_COPY_ON_INSTALL) to install folder
FOREACH(dir ${GOL_COPY_ON_INSTALL})
  #if source not exists than copy
  IF(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/${dir}")
      #copy important sub-folders from extension path (default PIConGPU parameter)
      IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
        INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/."
          DESTINATION "${CMAKE_INSTALL_PREFIX}/${dir}"
          PATTERN .svn EXCLUDE
        )
      ENDIF()
  ENDIF()
ENDFOREACH()

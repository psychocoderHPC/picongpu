#
# Copyright 2013-2015 Axel Huebl, Benjamin Schneider, Felix Schmitt,
#                     Heiko Burau, Rene Widera, Benjamin Worpitz
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################
# Required CMake version
################################################################################

cmake_minimum_required(VERSION 2.8.12.2)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project
################################################################################

PROJECT(PIConGPU)

# Set helper paths to find libraries and packages.
LIST(APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu" "$ENV{MPI_ROOT}" "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}" "$ENV{HDF5_ROOT}" "$ENV{VT_ROOT}")
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../thirdParty/cmake-modules")

################################################################################
# Configure Dependencies
################################################################################

#-------------------------------------------------------------------------------
# VampirTrace
#-------------------------------------------------------------------------------
OPTION(VAMPIR_ENABLE "Create PIConGPU with VampirTrace support" OFF)

# set filters: please do NOT use line breaks WITHIN the string!
set(VT_INST_FILE_FILTER
    "stl,usr/include,libgpugrid,vector_types.h,Vector.hpp,DeviceBuffer.hpp,DeviceBufferIntern.hpp,Buffer.hpp,StrideMapping.hpp,StrideMappingMethods.hpp,MappingDescription.hpp,AreaMapping.hpp,AreaMappingMethods.hpp,ExchangeMapping.hpp,ExchangeMappingMethods.hpp,DataSpace.hpp,Manager.hpp,Manager.tpp,Transaction.hpp,Transaction.tpp,TransactionManager.hpp,TransactionManager.tpp,Vector.tpp,Mask.hpp,ITask.hpp,EventTask.hpp,EventTask.tpp,StandardAccessor.hpp,StandardNavigator.hpp,HostBuffer.hpp,HostBufferIntern.hpp"
    CACHE STRING "VampirTrace: Files to exclude from instrumentation")
SET(VT_INST_FUNC_FILTER
    "vector,Vector,GPUGrid,execute,allocator,Task,Manager,Transaction,Mask,operator,DataSpace,PitchedBox,CudaEvent,new,getGridDim,GetCurrentDataSpaces,MappingDescription,getOffset,getParticlesBuffer,getDataSpace,getInstance"
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
    LIST(APPEND CMAKE_CXX_FLAGS "-finstrument-functions-exclude-FILE-LIST=${VT_INST_FILE_FILTER}")
    LIST(APPEND CMAKE_CXX_FLAGS "-finstrument-functions-exclude-function-LIST=${VT_INST_FUNC_FILTER}")

    # nvcc flags (rly necessary?)
    LIST(APPEND CUDA_NVCC_FLAGS
        -Xcompiler=-finstrument-functions,-finstrument-functions-exclude-FILE-LIST=\\\"${VT_INST_FILE_FILTER}\\\"
        -Xcompiler=-finstrument-functions-exclude-function-LIST=\\\"${VT_INST_FUNC_FILTER}\\\"
        -Xcompiler=-DVTRACE -Xcompiler=-I\\\"${VT_ROOT}/include/vampirtrace\\\"
        -v)

    # for manual instrumentation and hints that vampir is enabled in our code
    LIST(APPEND _PIC_COMPILE_DEFINITIONS_PRIVATE VTRACE)

    # titan work around: currently (5.14.4) the -D defines are not provided by -vt:showme-compile
    LIST(APPEND _PIC_COMPILE_DEFINITIONS_PRIVATE MPICH_IGNORE_CXX_SEEK)
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

    LIST(APPEND _PIC_FILES "${CMAKE_BINARY_DIR}/scorep_init.c")
ENDIF()

#-------------------------------------------------------------------------------
# Find PThreads
#-------------------------------------------------------------------------------
FIND_PACKAGE(Threads REQUIRED)
LIST(APPEND _PIC_LIBRARIES_PRIVATE ${CMAKE_THREAD_LIBS_INIT})

#-------------------------------------------------------------------------------
# Find OpenMP
#-------------------------------------------------------------------------------
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

#-------------------------------------------------------------------------------
# Find Boost.
#-------------------------------------------------------------------------------
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options regex system filesystem)
LIST(APPEND _PIC_INCLUDE_DIRECTORIES_PRIVATE ${Boost_INCLUDE_DIRS})
LIST(APPEND _PIC_LIBRARIES_PRIVATE ${Boost_LIBRARIES})

#-------------------------------------------------------------------------------
# Find PMacc.
#-------------------------------------------------------------------------------
SET("PMACC_ROOT" "${CMAKE_CURRENT_LIST_DIR}/../libPMacc" CACHE STRING  "The location of the PMacc library")

FIND_PACKAGE(PMacc REQUIRED)
LIST(APPEND _PIC_COMPILE_OPTIONS_PRIVATE ${PMacc_COMPILE_OPTIONS})
LIST(APPEND _PIC_COMPILE_DEFINITIONS_PRIVATE ${PMacc_COMPILE_DEFINITIONS})
LIST(APPEND _PIC_INCLUDE_DIRECTORIES_PRIVATE ${PMacc_INCLUDE_DIRS})
LIST(APPEND _PIC_LIBRARIES_PRIVATE ${PMacc_LIBRARIES})

#-------------------------------------------------------------------------------
# Find PNGwriter.
#-------------------------------------------------------------------------------
FIND_PACKAGE(PNGwriter)
IF(PNGwriter_FOUND)
    ADD_DEFINITIONS(${PNGwriter_DEFINITIONS})
    LIST(APPEND _PIC_COMPILE_DEFINITIONS_PRIVATE "PIC_ENABLE_PNG=1")
    LIST(APPEND _PIC_INCLUDE_DIRECTORIES_PRIVATE ${PNGwriter_INCLUDE_DIRS})
    LIST(APPEND _PIC_LIBRARIES_PRIVATE ${PNGwriter_LIBRARIES})
ENDIF(PNGwriter_FOUND)

#-------------------------------------------------------------------------------
# Find ADIOS.
#-------------------------------------------------------------------------------
#SET(ADIOS_USE_STATIC_LIBS ON) # force static linking
FIND_PACKAGE(ADIOS 1.6.0)
IF(ADIOS_FOUND)
    LIST(APPEND _PIC_COMPILE_DEFINITIONS_PRIVATE "ENABLE_ADIOS=1")
    IF(NOT (ADIOS_VERSION VERSION_LESS 1.7.0))
        MESSAGE(STATUS "Enable ADIOS data transforms")
        LIST(APPEND _PIC_COMPILE_DEFINITIONS_PRIVATE "ADIOS_TRANSFORMS=1")
    ENDIF(NOT (ADIOS_VERSION VERSION_LESS 1.7.0))

    LIST(APPEND _PIC_INCLUDE_DIRECTORIES_PRIVATE ${ADIOS_INCLUDE_DIRS})
    LIST(APPEND _PIC_LIBRARIES_PRIVATE ${ADIOS_LIBRARIES})
ENDIF(ADIOS_FOUND)

#-------------------------------------------------------------------------------
# Find libSplash (+ hdf5 due to required headers).
#-------------------------------------------------------------------------------
# prefer static libraries over shared ones (but do not force them)
SET(Splash_USE_STATIC_LIBS ON)
FIND_PACKAGE(Splash 1.2.4 COMPONENTS PARALLEL)
IF(Splash_FOUND)
    ADD_DEFINITIONS(${Splash_DEFINITIONS})
    LIST(APPEND _PIC_COMPILE_DEFINITIONS_PRIVATE "ENABLE_HDF5=1")
    LIST(APPEND _PIC_INCLUDE_DIRECTORIES_PRIVATE ${Splash_INCLUDE_DIRS})
    LIST(APPEND _PIC_LIBRARIES_PRIVATE ${Splash_LIBRARIES})
ENDIF(Splash_FOUND)

#-------------------------------------------------------------------------------
# Find mpiInfo.
#-------------------------------------------------------------------------------
FIND_PATH(
    MPI_INFO_DIR
    NAMES CMakeLists.txt
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../mpiInfo"
    DOC "path to mpiInfo")

ADD_SUBDIRECTORY(${MPI_INFO_DIR} "${CMAKE_CURRENT_BINARY_DIR}/build_mpiInfo")

#-------------------------------------------------------------------------------
# Find cuda_memtest.
#-------------------------------------------------------------------------------
IF(ALPAKA_ACC_GPU_CUDA_ENABLE)
    SET(SAME_NVCC_FLAGS_IN_SUBPROJECTS OFF)
    FIND_PATH(
        CUDA_MEMTEST_DIR
        NAMES CMakeLists.txt
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdParty/cuda_memtest"
        DOC "path to cuda_memtest")

    ADD_SUBDIRECTORY(${CUDA_MEMTEST_DIR} "${CMAKE_CURRENT_BINARY_DIR}/build_cuda_memtest")
ENDIF()

################################################################################
# PIConGPU.
################################################################################
SET(PIC_VERBOSE "1" CACHE STRING
    "Set verbosity level for PIConGPU (default is only physics output)")
LIST(APPEND _PIC_COMPILE_DEFINITIONS_PRIVATE "PIC_VERBOSE_LVL=${PIC_VERBOSE}")

OPTION(PIC_ENABLE_LIVE_VIEW "Enable Live View Plugin" OFF)
IF(PIC_ENABLE_LIVE_VIEW)
    LIST(APPEND _PIC_COMPILE_DEFINITIONS_PRIVATE "PIC_ENABLE_LIVE_VIEW=1")
ENDIF()

OPTION(PIC_ENABLE_INSITU_VOLVIS "Enable In Situ Volume Visualization" OFF)
IF(PIC_ENABLE_INSITU_VOLVIS)
    LIST(APPEND _PIC_COMPILE_DEFINITIONS_PRIVATE "ENABLE_INSITU_VOLVIS=1")
    SET(ALPAKA_CUDA_ARCH sm_35 CACHE STRING "Set the GPU architecture")

    SET(IceT_DIR "$ENV{ICET_ROOT}/lib")
    FIND_PACKAGE(IceT REQUIRED)
    LIST(APPEND _PIC_INCLUDE_DIRECTORIES_PRIVATE ${ICET_INCLUDE_DIRS})
    LIST(APPEND _PIC_LIBRARIES_PRIVATE ${ICET_CORE_LIBS} ${ICET_MPI_LIBS})
ENDIF()

#-------------------------------------------------------------------------------
# Check if PIC_EXTENSION_PATH is relative or absolute
#-------------------------------------------------------------------------------
FIND_PATH(
    PIC_EXTENSION_PATH
    NAMES "include/simulation_defines/param/componentsConfig.param"
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}"
    DOC "search PATH to extension folder"
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH)

SET(_PIC_EXTENSION_INCLUDE_DIR "${PIC_EXTENSION_PATH}/include")

# Prepend this include!
LIST(INSERT _PIC_INCLUDE_DIRECTORIES_PRIVATE 0 ${_PIC_EXTENSION_INCLUDE_DIR})

#-------------------------------------------------------------------------------
# PARAM (overwrite) defines
#-------------------------------------------------------------------------------
# TODO: Use the internal list that can be used with target commands instead of ADD_DEFINITIONS! (Remove all -D from all configuration files!)
IF(PARAM_OVERWRITES)
    FOREACH(param ${PARAM_OVERWRITES})
       ADD_DEFINITIONS(${param})
    ENDFOREACH(param)
ENDIF(PARAM_OVERWRITES)

################################################################################
# Compile & link PIConGPU
################################################################################

#-------------------------------------------------------------------------------
# Add all the include files in all recursive subdirectories and group them accordingly for MSVC projects.
#-------------------------------------------------------------------------------

FUNCTION(append_recursive_files_add_to_src_group_param_unitless_loader in_RootDir in_SrcGroupIgnorePrefix out_FilePathsListVariableName)
    SET(filePathsList)
    append_recursive_files_add_to_src_group("${in_RootDir}" "${in_SrcGroupIgnorePrefix}" "param" filePathsList)
    append_recursive_files_add_to_src_group("${in_RootDir}" "${in_SrcGroupIgnorePrefix}" "unitless" filePathsList)
    append_recursive_files_add_to_src_group("${in_RootDir}" "${in_SrcGroupIgnorePrefix}" "loader" filePathsList)
    SET(
        ${out_FilePathsListVariableName}
        "${${out_FilePathsListVariableName}}" "${filePathsList}"
        PARENT_SCOPE)
ENDFUNCTION()

append_recursive_files_add_to_src_group("${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_CURRENT_LIST_DIR}" "hpp" _PIC_FILES)
append_recursive_files_add_to_src_group("${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_CURRENT_LIST_DIR}" "tpp" _PIC_FILES)
append_recursive_files_add_to_src_group("${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_CURRENT_LIST_DIR}" "kernel" _PIC_FILES)
append_recursive_files_add_to_src_group("${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_CURRENT_LIST_DIR}" "def" _PIC_FILES)

append_recursive_files_add_to_src_group_param_unitless_loader("${_PIC_EXTENSION_INCLUDE_DIR}" "${_PIC_EXTENSION_INCLUDE_DIR}" _PIC_FILES)

# Add the full examples folder to the Project file (just for ease of usage, no compile relevance)
append_recursive_files_add_to_src_group_param_unitless_loader("${CMAKE_CURRENT_LIST_DIR}/../../examples" "${CMAKE_CURRENT_LIST_DIR}/../../" _PIC_FILES)

# Declare them as header files.
# Without this setting the '.def' files would not show up in the MSVC projects.
SET_SOURCE_FILES_PROPERTIES(
    ${_PIC_FILES}
    PROPERTIES HEADER_FILE_ONLY TRUE)

append_recursive_files_add_to_src_group("${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_CURRENT_LIST_DIR}" "cpp" _PIC_FILES)
append_recursive_files_add_to_src_group("${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_CURRENT_LIST_DIR}" "cu" _PIC_FILES)

# Force the main.cpp/main.cu file to be recognized as header and not be compiled so there wont be a second entry point depending on the usage of CUDA.
IF(ALPAKA_ACC_GPU_CUDA_ENABLE)
    SET_SOURCE_FILES_PROPERTIES(
        "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
        PROPERTIES HEADER_FILE_ONLY TRUE)
ELSE()
    SET_SOURCE_FILES_PROPERTIES(
        "${CMAKE_CURRENT_LIST_DIR}/main.cu"
        PROPERTIES HEADER_FILE_ONLY TRUE)
ENDIF()

#-------------------------------------------------------------------------------
# Add picongpu implementation specifics.
#-------------------------------------------------------------------------------
LIST(APPEND _PIC_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")
LIST(APPEND _PIC_INCLUDE_DIRECTORIES_PRIVATE ${_PIC_INCLUDE_DIR})

IF(WIN32)
    # On windows boost includes <windows.h> which by default defines a macro 'ERROR' which conflicts with ArgsParser::ERROR.
    LIST(APPEND _PIC_COMPILE_DEFINITIONS_PRIVATE "WIN32_LEAN_AND_MEAN" "VC_EXTRALEAN" "NOGDI")
    # On windows M_PI and other math defines are not defined by default in math.h or cmath because they are non-standard.
    LIST(APPEND _PIC_COMPILE_DEFINITIONS_PRIVATE "_USE_MATH_DEFINES")
ENDIF()

IF(MSVC)
    LIST(APPEND _PIC_COMPILE_OPTIONS_PRIVATE "/bigobj")
ENDIF()

SET(_PIC_DEFINITIONS ${_PIC_COMPILE_DEFINITIONS_PRIVATE})
list_add_prefix("-D" _PIC_DEFINITIONS)
LIST(APPEND _PIC_DEFINITIONS ${_PIC_COMPILE_OPTIONS_PRIVATE})
ADD_DEFINITIONS(
    ${_PIC_DEFINITIONS})
INCLUDE_DIRECTORIES(
    ${_PIC_INCLUDE_DIRECTORIES_PRIVATE})

#-------------------------------------------------------------------------------
# Create the executable target.
#-------------------------------------------------------------------------------
IF(ALPAKA_ACC_GPU_CUDA_ENABLE)
    CMAKE_POLICY(SET CMP0023 OLD)   # CUDA_ADD_EXECUTABLE calls TARGET_LINK_LIBRARIES without keywords.
    CUDA_ADD_EXECUTABLE(
        "picongpu"
        ${_PIC_FILES})
ELSE()
    ADD_EXECUTABLE(
        "picongpu"
        ${_PIC_FILES})
ENDIF()

TARGET_LINK_LIBRARIES(
    "picongpu"
    ${_PIC_LIBRARIES_PRIVATE})

#IF(SCOREP_ENABLE)
#    # Score-P libraries must be linked after the object to prevent symbols
#    # from being stripped on Titan
#    target_link_libraries(picongpu ${SCOREP_LIBFLAGS})
#ENDIF(SCOREP_ENABLE)

################################################################################
# Install PIConGPU
################################################################################

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
ENDIF()

SET(PIC_COPY_ON_INSTALL "include/simulation_defines" "submit" CACHE LIST
    "Folder which is copied to installation path during install" )

install(TARGETS picongpu
         RUNTIME DESTINATION bin)

#file(GLOB scripts_to_copy "${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.sh")
#foreach(f ${scripts_to_copy})
#   GET_FILENAME_COMPONENT(file_name ${f} NAME CACHE)
#   install(FILES "${f}" DESTINATION bin PERMISSIONS OWNER_EXECUTE OWNER_READ
#           OWNER_WRITE GROUP_READ GROUP_EXECUTE)
#endforeach(f)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scripts" DESTINATION bin
    FILES_MATCHING PATTERN "*"
    PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_READ GROUP_EXECUTE
    PATTERN .svn EXCLUDE
)

# If the installation prefix does not equal extension path, check if folder must be copied.
# If there is no include folder in installation prefix, also copy all missing folders.
IF( (NOT "${CMAKE_INSTALL_PREFIX}" STREQUAL "${PIC_EXTENSION_PATH}") OR
    (NOT EXISTS "${CMAKE_INSTALL_PREFIX}/include"))

    #copy all important subfolders to install folder
    foreach(dir ${PIC_COPY_ON_INSTALL})

      #if source not exists than copy
      IF(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/${dir}")
          #copy important subfolders from extension path (default PIConGPU parameter)
          IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
            install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/."
              DESTINATION "${CMAKE_INSTALL_PREFIX}/${dir}"
              PATTERN .svn EXCLUDE
              PATTERN .git EXCLUDE
            )
          ENDIF()

          #copy important subfolders from extension path (from extension path)
          IF(EXISTS "${PIC_EXTENSION_PATH}/${dir}")
            install(DIRECTORY "${PIC_EXTENSION_PATH}/${dir}/."
               DESTINATION "${CMAKE_INSTALL_PREFIX}/${dir}"
               PATTERN .svn EXCLUDE
               PATTERN .git EXCLUDE
            )
          ENDIF()
      ENDIF()

    endforeach()
ENDIF()

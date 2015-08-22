################################################################################
# Copyright 2013-2015 Benjamin Worpitz, Rene Widera, Felix Schmitt,
#                     Heiko Burau, Axel Huebl
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
# RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE
# USE OR PERFORMANCE OF THIS SOFTWARE.
################################################################################

################################################################################
# Required cmake version.
################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12.2)

################################################################################
# PMacc.
################################################################################

# Return values.
UNSET(PMacc_FOUND)
UNSET(PMacc_VERSION)
UNSET(PMacc_COMPILE_OPTIONS)
UNSET(PMacc_COMPILE_DEFINITIONS)
UNSET(PMacc_DEFINITIONS)
UNSET(PMacc_INCLUDE_DIR)
UNSET(PMacc_INCLUDE_DIRS)
UNSET(PMacc_LIBRARY)
UNSET(PMacc_LIBRARIES)

# Internal usage.
UNSET(_PMACC_FOUND)
UNSET(_PMACC_VERSION)
UNSET(_PMACC_COMPILE_OPTIONS_PUBLIC)
UNSET(_PMACC_COMPILE_DEFINITIONS_PUBLIC)
UNSET(_PMACC_INCLUDE_DIR)
UNSET(_PMACC_INCLUDE_DIRECTORIES_PUBLIC)
UNSET(_PMACC_LINK_LIBRARIES_PUBLIC)
UNSET(_PMACC_FILES_HEADER)
UNSET(_PMACC_FILES_SOURCE)
UNSET(_PMACC_FILES_OTHER)

#-------------------------------------------------------------------------------
# Directory of this file.
#-------------------------------------------------------------------------------
SET(_PMACC_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

# Normalize the path (e.g. remove ../)
GET_FILENAME_COMPONENT(_PMACC_ROOT_DIR "${_PMACC_ROOT_DIR}" ABSOLUTE)

#-------------------------------------------------------------------------------
# Set found to true initially and SET it on false IF a required dependency is missing.
#-------------------------------------------------------------------------------
SET(_PMACC_FOUND TRUE)

#-------------------------------------------------------------------------------
# Common.
#-------------------------------------------------------------------------------
# own modules for find_packages
LIST(APPEND CMAKE_MODULE_PATH "${_PMACC_ROOT_DIR}/../../thirdParty/cmake-modules")

#-------------------------------------------------------------------------------
# Options.
#-------------------------------------------------------------------------------
SET(PMACC_VERBOSE "0" CACHE STRING "Set verbosity level for libPMacc")
LIST(APPEND _PMACC_COMPILE_DEFINITIONS_PUBLIC "PMACC_VERBOSE_LVL=${PMACC_VERBOSE}")

OPTION(PMACC_BLOCKING_KERNEL "Activate checks for every kernel call and synchronize after every kernel call" OFF)
IF(PMACC_BLOCKING_KERNEL)
    LIST(APPEND _PMACC_COMPILE_DEFINITIONS_PUBLIC "PMACC_SYNC_KERNEL=1")
ENDIF(PMACC_BLOCKING_KERNEL)

#-------------------------------------------------------------------------------
# Find alpaka
# NOTE: Do this first, because it declares `list_add_prefix` and `append_recursive_files_add_to_src_group` used later on.
#-------------------------------------------------------------------------------
SET("ALPAKA_ROOT" "${_PMACC_ROOT_DIR}/../../../alpaka" CACHE STRING  "The location of the alpaka library")
FIND_PACKAGE(alpaka)
IF(NOT alpaka_FOUND)
    MESSAGE(WARNING "Required PMacc dependency alpaka could not be found!")
    SET(_PMACC_FOUND FALSE)

ELSE()
    LIST(APPEND _PMACC_COMPILE_OPTIONS_PUBLIC ${alpaka_COMPILE_OPTIONS})
    LIST(APPEND _PMACC_COMPILE_DEFINITIONS_PUBLIC ${alpaka_COMPILE_DEFINITIONS})
    LIST(APPEND _PMACC_INCLUDE_DIRECTORIES_PUBLIC ${alpaka_INCLUDE_DIRS})
    LIST(APPEND _PMACC_LINK_LIBRARIES_PUBLIC ${alpaka_LIBRARIES})
ENDIF()

#-------------------------------------------------------------------------------
# Find MPI
#-------------------------------------------------------------------------------
FIND_PACKAGE(MPI)
IF(NOT MPI_CXX_FOUND)
    MESSAGE(WARNING "Required PMacc dependency MPI could not be found!")
    SET(_PMACC_FOUND FALSE)

ELSE()
    LIST(APPEND _PMACC_COMPILE_OPTIONS_PUBLIC  ${MPI_CXX_COMPILE_FLAGS})
    LIST(APPEND _PMACC_INCLUDE_DIRECTORIES_PUBLIC ${MPI_CXX_INCLUDE_PATH})

    SET(_PMACC_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
    list_add_prefix("general;" _PMACC_MPI_LIBRARIES)
    LIST(APPEND _PMACC_LINK_LIBRARIES_PUBLIC ${_PMACC_MPI_LIBRARIES})
    UNSET(_PMACC_MPI_LIBRARIES)
ENDIF()

#-------------------------------------------------------------------------------
# Find Boost
#-------------------------------------------------------------------------------
FIND_PACKAGE(Boost 1.49.0)
IF(NOT Boost_FOUND)
    MESSAGE(WARNING "Required PMacc dependency Boost could not be found!")
    SET(_PMACC_FOUND FALSE)

ELSE()
    LIST(APPEND _PMACC_INCLUDE_DIRECTORIES_PUBLIC ${Boost_INCLUDE_DIRS})
ENDIF()

#-------------------------------------------------------------------------------
# Find mallocMC
#-------------------------------------------------------------------------------
FIND_PACKAGE(mallocMC 2.1.0 QUIET)
IF(NOT mallocMC_FOUND)
    MESSAGE(STATUS "Trying to use mallocMC from thirdParty/ directory")
    SET(MALLOCMC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdParty/mallocMC")
    FIND_PACKAGE(mallocMC 2.1.0 QUIET)
    IF(NOT mallocMC_FOUND)
        MESSAGE(WARNING "Required PMacc dependency mallocMC could not be found!")
        SET(_PMACC_FOUND FALSE)
    ENDIF()
ENDIF()

IF(mallocMC_FOUND)
    LIST(APPEND _PMACC_INCLUDE_DIRECTORIES_PUBLIC ${mallocMC_INCLUDE_DIRS})
    # TODO: Compile options or compile definitions?
    LIST(APPEND _PMACC_COMPILE_OPTIONS_PUBLIC ${mallocMC_DEFINITIONS})
    LIST(APPEND _PMACC_LINK_LIBRARIES_PUBLIC ${mallocMC_LIBRARIES})
ENDIF()

#-------------------------------------------------------------------------------
# Compiler settings.
#-------------------------------------------------------------------------------
IF(MSVC)
    # Empty append to define it IF it does not already exist.
    LIST(APPEND _PMACC_COMPILE_OPTIONS_PUBLIC)
ELSE()
    # GNU
    IF(CMAKE_COMPILER_IS_GNUCXX)
        LIST(APPEND _PMACC_COMPILE_OPTIONS_PUBLIC "-Wall")
        LIST(APPEND _PMACC_COMPILE_OPTIONS_PUBLIC "-Wextra")
        LIST(APPEND _PMACC_COMPILE_OPTIONS_PUBLIC "-Wno-unknown-pragmas")
        LIST(APPEND _PMACC_COMPILE_OPTIONS_PUBLIC "-Wno-unused-parameter")
        LIST(APPEND _PMACC_COMPILE_OPTIONS_PUBLIC "-Wno-unused-local-typedefs")
        LIST(APPEND _PMACC_COMPILE_OPTIONS_PUBLIC "-Wno-attributes")
        LIST(APPEND _PMACC_COMPILE_OPTIONS_PUBLIC "-Wno-reorder")
        LIST(APPEND _PMACC_COMPILE_OPTIONS_PUBLIC "-Wno-sign-compare")
    # ICC
    ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        LIST(APPEND _PMACC_COMPILE_OPTIONS_PUBLIC "-Wall")
        LIST(APPEND _PMACC_COMPILE_DEFINITIONS_PUBLIC "BOOST_NO_FENV_H")
    # PGI
    ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "PGI")
        LIST(APPEND _PMACC_COMPILE_OPTIONS_PUBLIC "-Minform=inform")
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# PMacc.
#-------------------------------------------------------------------------------
SET(_PMACC_INCLUDE_DIR "${_PMACC_ROOT_DIR}/include")
LIST(APPEND _PMACC_INCLUDE_DIRECTORIES_PUBLIC ${_PMACC_INCLUDE_DIR})
SET(_PMACC_SUFFIXED_INCLUDE_DIR "${_PMACC_INCLUDE_DIR}")
#SET(_PMACC_SUFFIXED_INCLUDE_DIR "${_PMACC_INCLUDE_DIR}/PMacc")

SET(_PMACC_LINK_LIBRARY)
LIST(APPEND _PMACC_LINK_LIBRARIES_PUBLIC ${_PMACC_LINK_LIBRARY})

SET(_PMACC_FILES_OTHER "${_PMACC_ROOT_DIR}/FindPMacc.cmake" "${_PMACC_ROOT_DIR}/PMaccConfig.cmake")

# Add all the include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_PMACC_SUFFIXED_INCLUDE_DIR}" "${_PMACC_SUFFIXED_INCLUDE_DIR}" "hpp" _PMACC_FILES_HEADER)
append_recursive_files_add_to_src_group("${_PMACC_SUFFIXED_INCLUDE_DIR}" "${_PMACC_SUFFIXED_INCLUDE_DIR}" "h" _PMACC_FILES_HEADER)
append_recursive_files_add_to_src_group("${_PMACC_SUFFIXED_INCLUDE_DIR}" "${_PMACC_SUFFIXED_INCLUDE_DIR}" "tpp" _PMACC_FILES_HEADER)
append_recursive_files_add_to_src_group("${_PMACC_SUFFIXED_INCLUDE_DIR}" "${_PMACC_SUFFIXED_INCLUDE_DIR}" "def" _PMACC_FILES_HEADER)
append_recursive_files_add_to_src_group("${_PMACC_SUFFIXED_INCLUDE_DIR}" "${_PMACC_SUFFIXED_INCLUDE_DIR}" "kernel" _PMACC_FILES_HEADER)

#append_recursive_files_add_to_src_group("${_PMACC_SUFFIXED_INCLUDE_DIR}" "${_PMACC_SUFFIXED_INCLUDE_DIR}" "cpp" _PMACC_FILES_SOURCE)

#-------------------------------------------------------------------------------
# Target.
#-------------------------------------------------------------------------------
IF(NOT TARGET PMacc)

    ADD_LIBRARY(
        "PMacc"
        ${_PMACC_FILES_HEADER} ${_PMACC_FILES_SOURCE} ${_PMACC_FILES_OTHER})

    # Even if there are no sources CMAKE has to know the language.
    SET_TARGET_PROPERTIES("PMacc" PROPERTIES LINKER_LANGUAGE CXX)

    # Compile options.
    MESSAGE(STATUS "_PMACC_COMPILE_OPTIONS_PUBLIC: ${_PMACC_COMPILE_OPTIONS_PUBLIC}")
    LIST(
        LENGTH
        _PMACC_COMPILE_OPTIONS_PUBLIC
        _PMACC_COMPILE_OPTIONS_PUBLIC_LENGTH)
    IF("${_PMACC_COMPILE_OPTIONS_PUBLIC_LENGTH}")
        TARGET_COMPILE_OPTIONS(
            "PMacc"
            PUBLIC ${_PMACC_COMPILE_OPTIONS_PUBLIC})
    ENDIF()

    # Compile definitions.
    MESSAGE(STATUS "_PMACC_COMPILE_DEFINITIONS_PUBLIC: ${_PMACC_COMPILE_DEFINITIONS_PUBLIC}")
    LIST(
        LENGTH
        _PMACC_COMPILE_DEFINITIONS_PUBLIC
        _PMACC_COMPILE_DEFINITIONS_PUBLIC_LENGTH)
    IF("${_PMACC_COMPILE_DEFINITIONS_PUBLIC_LENGTH}")
        TARGET_COMPILE_DEFINITIONS(
            "PMacc"
            PUBLIC ${_PMACC_COMPILE_DEFINITIONS_PUBLIC})
    ENDIF()

    # Include directories.
    MESSAGE(STATUS "_PMACC_INCLUDE_DIRECTORIES_PUBLIC: ${_PMACC_INCLUDE_DIRECTORIES_PUBLIC}")
    LIST(
        LENGTH
        _PMACC_INCLUDE_DIRECTORIES_PUBLIC
        _PMACC_INCLUDE_DIRECTORIES_PUBLIC_LENGTH)
    IF("${_PMACC_INCLUDE_DIRECTORIES_PUBLIC_LENGTH}")
        TARGET_INCLUDE_DIRECTORIES(
            "PMacc"
            PUBLIC ${_PMACC_INCLUDE_DIRECTORIES_PUBLIC})
    ENDIF()

    # Link libraries.
    MESSAGE(STATUS "_PMACC_LINK_LIBRARIES_PUBLIC: ${_PMACC_LINK_LIBRARIES_PUBLIC}")
    #LIST(
    #    LENGTH
    #    _PMACC_LINK_LIBRARIES_PUBLIC
    #    _PMACC_LINK_LIBRARIES_PUBLIC_LENGTH)
    #IF("${_PMACC_LINK_LIBRARIES_PUBLIC_LENGTH}")
        TARGET_LINK_LIBRARIES(
            "PMacc"
            PUBLIC alpaka ${_PMACC_LINK_LIBRARIES_PUBLIC})
    #ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Find PMacc version.
#-------------------------------------------------------------------------------
# FIXME: Add a version.hpp
SET(_PMACC_VERSION "0.1.0")

#-------------------------------------------------------------------------------
# Set return values.
#-------------------------------------------------------------------------------
SET(PMacc_VERSION ${_PMACC_VERSION})
SET(PMacc_COMPILE_OPTIONS ${_PMACC_COMPILE_OPTIONS_PUBLIC})
SET(PMacc_COMPILE_DEFINITIONS ${_PMACC_COMPILE_DEFINITIONS_PUBLIC})
# Add '-D' to the definitions
SET(PMacc_DEFINITIONS ${_PMACC_COMPILE_DEFINITIONS_PUBLIC})
list_add_prefix("-D" PMacc_DEFINITIONS)
# Add the compile options to the definitions.
LIST(APPEND PMacc_DEFINITIONS ${_PMACC_COMPILE_OPTIONS_PUBLIC})
SET(PMacc_INCLUDE_DIR ${_PMACC_INCLUDE_DIR})
SET(PMacc_INCLUDE_DIRS ${_PMACC_INCLUDE_DIRECTORIES_PUBLIC})
SET(PMacc_LIBRARY ${_PMACC_LINK_LIBRARY})
SET(PMacc_LIBRARIES ${_PMACC_LINK_LIBRARIES_PUBLIC})

# Unset already SET variables IF not found.
IF(NOT _PMACC_FOUND)
    UNSET(PMacc_FOUND)
    UNSET(PMacc_VERSION)
    UNSET(PMacc_COMPILE_OPTIONS)
    UNSET(PMacc_COMPILE_DEFINITIONS)
    UNSET(PMacc_DEFINITIONS)
    UNSET(PMacc_INCLUDE_DIR)
    UNSET(PMacc_INCLUDE_DIRS)
    UNSET(PMacc_LIBRARY)
    UNSET(PMacc_LIBRARIES)

    UNSET(_PMACC_FOUND)
    UNSET(_PMACC_COMPILE_OPTIONS_PUBLIC)
    UNSET(_PMACC_COMPILE_DEFINITIONS_PUBLIC)
    UNSET(_PMACC_INCLUDE_DIR)
    UNSET(_PMACC_INCLUDE_DIRECTORIES_PUBLIC)
    UNSET(_PMACC_LINK_LIBRARY)
    UNSET(_PMACC_LINK_LIBRARIES_PUBLIC)
    UNSET(_PMACC_FILES_HEADER)
    UNSET(_PMACC_FILES_SOURCE)
    UNSET(_PMACC_FILES_OTHER)
    UNSET(_PMACC_VERSION)
ELSE()
    # Make internal variables advanced options in the GUI.
    MARK_AS_ADVANCED(
        PMacc_INCLUDE_DIR
        PMacc_LIBRARY
        _PMACC_FOUND
        _PMACC_COMPILE_OPTIONS_PUBLIC
        _PMACC_COMPILE_DEFINITIONS_PUBLIC
        _PMACC_INCLUDE_DIR
        _PMACC_INCLUDE_DIRECTORIES_PUBLIC
        _PMACC_LINK_LIBRARY
        _PMACC_LINK_LIBRARIES_PUBLIC
        _PMACC_FILES_HEADER
        _PMACC_FILES_SOURCE
        _PMACC_FILES_OTHER
        _PMACC_VERSION)
ENDIF()

###############################################################################
# FindPackage options
###############################################################################

# Handles the REQUIRED, QUIET and version-related arguments for FIND_PACKAGE.
# NOTE: We do not check for PMacc_LIBRARIES and PMacc_DEFINITIONS because they can be empty.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(
    "PMacc"
    FOUND_VAR PMacc_FOUND
    REQUIRED_VARS PMacc_INCLUDE_DIR
    VERSION_VAR PMacc_VERSION)

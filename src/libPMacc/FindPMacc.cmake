#.rst:
# FindPMacc
# ----------
#
# TODO: short description
# TODO: https://github.com/ComputationalRadiationPhysics/PMacc
#
# Finding and Using PMacc
# ^^^^^^^^^^^^^^^^^^^^^
#
# .. code-block:: cmake
#
#   FIND_PACKAGE(PMacc
#     [version] [EXACT]     # Minimum or EXACT version, e.g. 1.0.0
#     [REQUIRED]            # Fail with an error if PMacc or a required
#                           # component is not found
#     [QUIET]               # Do not warn if this module was not found
#     [COMPONENTS <...>]    # Compiled in components: ignored
#   )
#   TARGET_LINK_LIBRARIES(<target> PUBLIC PMacc)
#
# To provide a hint to this module where to find the alpaka installation,
# set the ALPAKA_ROOT variable.
#
# This module requires Boost. Make sure to provide a valid install of it
# under the environment variable BOOST_ROOT.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# - ``PMacc_FOUND``
#   TRUE if PMacc found a working install.
# - ``PMacc_VERSION``
#   Version in format Major.Minor.Patch
# - ``PMacc_COMPILE_OPTIONS``
#   Compiler flags.
# - ``PMacc_COMPILE_DEFINITIONS``
#   Compiler defines.
# - ``PMacc_DEFINITIONS``
#   Deprecated old compiler definitions. Combination of PMacc_COMPILE_OPTIONS and PMacc_COMPILE_DEFINITIONS prefixed with "-D".
# - ``PMacc_INCLUDE_DIRS``
#   Include directories for the PMacc headers.
# - ``PMacc_LIBRARIES``
#   PMacc libraries.
#
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the :prop_tgt:`IMPORTED` target ``PMacc``, if PMacc has
# been found.
#


################################################################################
# Copyright 2015 Benjamin Worpitz
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

FIND_PATH(
    _PMACC_ROOT_DIR
    NAMES "include/types.h"
    HINTS "${PMACC_ROOT}" ENV PMACC_ROOT
    DOC "PMacc ROOT location")

IF(_PMACC_ROOT_DIR)
    INCLUDE("${_PMACC_ROOT_DIR}/PMaccConfig.cmake")
ELSE()
    MESSAGE(FATAL_ERROR "PMacc could not be found!")
ENDIF()

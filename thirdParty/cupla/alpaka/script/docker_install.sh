#!/bin/bash

#
# Copyright 2017-2019 Benjamin Worpitz
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/set.sh

# runtime and compile time options
ALPAKA_DOCKER_ENV_LIST=()
ALPAKA_DOCKER_ENV_LIST+=("--env" "CC=${CC}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "CXX=${CXX}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "TRAVIS_OS_NAME=${TRAVIS_OS_NAME}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_DOCKER_BASE_IMAGE_NAME=${ALPAKA_CI_DOCKER_BASE_IMAGE_NAME}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_ANALYSIS=${ALPAKA_CI_ANALYSIS}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_BOOST_BRANCH=${ALPAKA_CI_BOOST_BRANCH}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "BOOST_ROOT=${BOOST_ROOT}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_BOOST_LIB_DIR=${ALPAKA_CI_BOOST_LIB_DIR}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CLANG_DIR=${ALPAKA_CI_CLANG_DIR}")
if [ ! -z "${ALPAKA_CI_CLANG_VER+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CLANG_VER=${ALPAKA_CI_CLANG_VER}")
fi
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_STDLIB=${ALPAKA_CI_STDLIB}")
if [ ! -z ${ALPAKA_CI_CLANG_LIBSTDCPP_VERSION+x} ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CLANG_LIBSTDCPP_VERSION=${ALPAKA_CI_CLANG_LIBSTDCPP_VERSION}")
fi
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CMAKE_VER=${ALPAKA_CI_CMAKE_VER}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CMAKE_DIR=${ALPAKA_CI_CMAKE_DIR}")
if [ ! -z "${ALPAKA_CI_GCC_VER+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_GCC_VER=${ALPAKA_CI_GCC_VER}")
fi
if [ ! -z "${ALPAKA_CI_SANITIZERS+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_SANITIZERS=${ALPAKA_CI_SANITIZERS}")
fi
if [ ! -z "${ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE}")
fi
if [ ! -z "${ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE}")
fi
if [ ! -z "${ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE}")
fi
if [ ! -z "${ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=${ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE}")
fi
if [ ! -z "${ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE}")
fi
if [ ! -z "${ALPAKA_ACC_CPU_BT_OMP4_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_BT_OMP4_ENABLE=${ALPAKA_ACC_CPU_BT_OMP4_ENABLE}")
fi
if [ ! -z "${ALPAKA_ACC_GPU_CUDA_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_GPU_CUDA_ENABLE=${ALPAKA_ACC_GPU_CUDA_ENABLE}")
fi
if [ ! -z "${ALPAKA_ACC_GPU_HIP_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_GPU_HIP_ENABLE=${ALPAKA_ACC_GPU_HIP_ENABLE}")
fi
if [ ! -z "${ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE=${ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE}")
fi
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_INSTALL_CUDA=${ALPAKA_CI_INSTALL_CUDA}")
if [ "${ALPAKA_CI_INSTALL_CUDA}" == "ON" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CUDA_DIR=${ALPAKA_CI_CUDA_DIR}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CUDA_VERSION=${ALPAKA_CUDA_VERSION}")
    if [ ! -z "${ALPAKA_CUDA_COMPILER+x}" ]
    then
        ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CUDA_COMPILER=${ALPAKA_CUDA_COMPILER}")
    fi
fi
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_INSTALL_HIP=${ALPAKA_CI_INSTALL_HIP}")
if [ "${ALPAKA_CI_INSTALL_HIP}" == "ON" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_HIP_ROOT_DIR=${ALPAKA_CI_HIP_ROOT_DIR}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_HIP_BRANCH=${ALPAKA_CI_HIP_BRANCH}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_HIP_PLATFORM=${ALPAKA_HIP_PLATFORM}")
fi
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_INSTALL_TBB=${ALPAKA_CI_INSTALL_TBB}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_INSTALL_FIBERS=${ALPAKA_CI_INSTALL_FIBERS}")

docker run -v "$(pwd)":"$(pwd)" -w "$(pwd)" "${ALPAKA_DOCKER_ENV_LIST[@]}" "${ALPAKA_CI_DOCKER_BASE_IMAGE_NAME}" /bin/bash ./script/install.sh

ALPAKA_DOCKER_CONTAINER_NAME=$(docker ps -l -q)
docker commit "${ALPAKA_DOCKER_CONTAINER_NAME}" "${ALPAKA_CI_DOCKER_IMAGE_NAME}"

# delete the container and the base image to save disc space
docker stop "${ALPAKA_DOCKER_CONTAINER_NAME}"
docker rm "${ALPAKA_DOCKER_CONTAINER_NAME}"
docker rmi "${ALPAKA_CI_DOCKER_BASE_IMAGE_NAME}"

docker images

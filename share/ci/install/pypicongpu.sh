#!/bin/bash

# This file is part of the PIConGPU.
# Copyright 2023-2023 PIConGPU contributors
# Authors: Simeon Ehrig
# License: GPLv3+

# - the script installs a Python environment
# - generates a modified requirements.txt depending of the environment variables for pypicongpu
# - install the dependencies and runs the quick tests

set -e
set -o pipefail

function script_error {
    echo -e "\e[31mERROR: ${1}\e[0m"
    exit 1
}

export PICSRC=$CI_PROJECT_DIR
export PATH=$PATH:$PICSRC/bin
export PYTHONPATH=$PICSRC/lib/python:$PYTHONPATH
export PIC_EXAMPLES=$PICSRC/share/picongpu/examples

cd $CI_PROJECT_DIR

# use miniconda as python environment
apt update && apt install -y wget
cd /tmp/
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod u+x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3
export PATH=/miniconda3/bin:$PATH
conda --version
source /miniconda3/etc/profile.d/conda.sh

cd $CI_PROJECT_DIR
# generates modified requirements.txt
conda create -n pypicongpu python=${PYTHON_VERSION}
conda activate pypicongpu
python3 --version
MODIFIED_REQUIREMENT_TXT=$CI_PROJECT_DIR/lib/python/picongpu/modified_requirements.txt
python3 $CI_PROJECT_DIR/share/ci/install/requirements_txt_modifier.py $CI_PROJECT_DIR/lib/python/picongpu/requirements.txt $MODIFIED_REQUIREMENT_TXT

echo "modified_requirements.txt: "
cat $MODIFIED_REQUIREMENT_TXT
echo ""

# run quick tests
pip3 install --break-system-packages -r $MODIFIED_REQUIREMENT_TXT
cd $CI_PROJECT_DIR/test/python/picongpu
python3 -m quick

# executing the compiling tests is optional
# for the compiling test we need: cmake, boost and openmpi
# openmpi is available without extra work
if [ ! -z ${PYTHON_COMPILING_TEST+x} ]; then
    export PIC_BACKEND=omp2b
    # setup cmake
    if [ ! -z ${CMAKE_VERSION+x} ]; then
        if agc-manager -e cmake@${CMAKE_VERSION} ; then
            export PATH=$(agc-manager -b cmake@${CMAKE_VERSION})/bin:$PATH
        else
            script_error "No implementation to install cmake ${CMAKE_VERSION}"
        fi
    else
        script_error "CMAKE_VERSION is not defined"
    fi

    # setup boost
    if [ ! -z ${BOOST_VERSION+x} ]; then
        if agc-manager -e boost@${BOOST_VERSION} ; then
            export CMAKE_PREFIX_PATH=$(agc-manager -b boost@${BOOST_VERSION}):$CMAKE_PREFIX_PATH
        else
            script_error "No implementation to install boost ${BOOST_VERSION}"
        fi
    else
        script_error "BOOST_VERSION is not defined"
    fi

    # execute the compiling test
    python3 -m compiling -v
fi

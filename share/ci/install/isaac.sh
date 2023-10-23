#!/bin/bash

set -e
set -o pipefail

# merge the PR to the latest version of the destination branch

cd $CI_PROJECT_DIR

export GLM_ROOT=/opt/glm/0.9.9.9-dev
export CMAKE_PREFIX_PATH=$GLM_ROOT:$CMAKE_PREFIX_PATH
git clone https://github.com/g-truc/glm.git
cd glm
git checkout 47585fde0c49fa77a2bf2fb1d2ead06999fd4b6e
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=$GLM_ROOT -DGLM_TEST_ENABLE=OFF
make install

cd $CI_PROJECT_DIR
git clone https://github.com/ComputationalRadiationPhysics/isaac.git
cd isaac
git checkout 014dc16e9ab192821ba941e1ceb0c313d94b71fe
mkdir build_isaac
cd build_isaac
cmake ../lib/ -DCMAKE_INSTALL_PREFIX=$ISAAC_ROOT
make install
cd ..

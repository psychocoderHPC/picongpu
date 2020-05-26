# Building configuring for the OpenACC backend

## CMake Basics

Alpaka requires cmake version >= 3.15.

In the root of the alpaka dir, run:
```bash
mkdir build
cd build
```

## Configuring Using CMake

In the build directory, invoke cmake to configure. Use the options below to
enable only the OpenACC backend.

```bash
cmake .. \
  -DALPAKA_ACC_ANY_BT_OACC_ENABLE=on \
  -DALPAKA_ACC_CPU_BT_OMP4_ENABLE=off \
  -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=on \
  -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=off \
  -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE=off \
  -DALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE=off \
  -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE=off \
  -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE=off \
  -DALPAKA_ACC_GPU_CUDA_ENABLE=off \
  -DALPAKA_ACC_GPU_HIP_ENABLE=off \
```
All other backends are disabled for faster compilation/testing and reduced
environment requirements. The cmake package OpenACC is used to detect the
required OpenACC flags for the compiler. Additional flags can be added, e.g:
- gcc, target x86:
  ```bash
    -DCMAKE_CXX_FLAGS="-foffload=disable -fopenacc"
  ```
  As of gcc 9.2 no test will compile if the nvptx backend is enabled. If cmake
  fails to set the `-fopenacc` flag, it can be set manually.
- pgi, target tesla:
  ```bash
    -DCMAKE_CXX_FLAGS="-acc -ta=tesla -Minfo"
  ```

## Test targets

### helloWorld

```bash
make helloWorld
./examples/helloWorld/helloWorld
```
The output should end with
```
[z:3, y:7, x:15][linear:511] Hello World
```
Numbers can vary when teams are executed in parallel: 512 teams, with one worker
each are started in a 3d grid. Each reports its grid coordinates and linear
index.

|compiler|compile status|target|run status|
|---|---|---|---|
|GCC 10(dev)| ok|x86|ok|
|PGI 19.10| ICE|tesla|--|

### vectorAdd

```bash
make vectorAdd
./examples/vectorAdd/vectorAdd
```
The output should end with
```
Execution results correct!
```

|compiler|compile status|target|run status|
|---|---|---|---|
|GCC 10(dev)| ok|x86|ok|
|PGI 19.10| fail (1) |tesla|--|

1. 
  ```
  PGCC-W-0155-External and Static variables are not supported in acc routine - _T140319056362856_40208
  (~alpaka/example/vectorAdd/src/vectorAdd.cpp: 47)
  ```
  The indicated line is a parameter of a function template called in an acc
  parallel region. The type in this instance is `std::uint32_t const * const`.
  
## Building and Running all tests

```bash
make
ctest
```

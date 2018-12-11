#!/bin/bash

smiOutput=`nvidia-smi topo -m `
numCores=`cat /proc/cpuinfo | grep processor| tail -n 1 | cut -d":" -f 2`
let numCores=numCores+1
numCPUSockets=`numactl --hardware | grep available | cut -d" " -f 2`
numGPUs=`echo "$smiOutput" | sed  -n '2,${p}' | grep GPU | wc -l`
let numGPUsPerSocket=numGPUs/numCPUSockets

ibOffsets=`echo "$smiOutput" | head -n1 | awk '{for(i=1;i<NF;++i){ if($i~/mlx/) printf("%i ",i+1);} }'`

if [ -n "$OMPI_COMM_WORLD_LOCAL_RANK" ] ; then
  
  let localRank=$OMPI_COMM_WORLD_LOCAL_RANK

  let cpuSocket=localRank%numCPUSockets
  let socketGPUOffset=cpuSocket*numGPUsPerSocket
  let gpuId=localRank/numCPUSockets+socketGPUOffset

  gpuConnections=`echo "$smiOutput" | sed  -n "$((gpuId+2))"p`

  for conn in $ibOffsets
  do
    sockType=`echo "$gpuConnections" | awk -v i=$conn '{print $i}'`
    if [ $sockType != "SOC" ] ; then
      ibName=`echo "$smiOutput"| head -n1 | awk -v n=$conn '{printf("%s",$(n-1))}'`
      ibDevices+="$ibName "
    fi
  done

  ibDevices=`echo $ibDevices | tr " " ","`

  export MPI_LOCAL_RANK=$localRank
  if [ -n "$ibDevices" ] ; then
    #echo "enable num dma"
    export OMPI_MCA_btl_openib_if_include=$ibDevices
    export OMPI_MCA_btl_openib_want_cuda_gdr=1
#    export OMPI_MCA_btl_openib_cuda_rdma_limit=30000
#    export OMPI_MCA_coll_fca_enable=0
#    export OMPI_MCA_btl_openib_cuda_rdma_limit=1342177280 
    #export OMPI_MCA_btl_smcuda_free_list_num=2048
    #export OMPI_MCA_btl_smcuda_free_list_inc=1024
    #export OMPI_MCA_btl_smcuda_fifo_lazy_free=1024
#     export OMPI_MCA_btl_openib_cuda_rdma_limit=64
    export OMPI_MCA_btl_smcuda_use_cuda_ipc=1
    export OMPI_MCA_btl_smcuda_use_cuda_ipc_same_gpu=1
#    export OMPI_MCA_mpi_common_cuda_verbose=100
#    export MPI_GPU_DIRECT=1
  fi
  export OMP_NUM_THREADS="$((numCores/numCPUSockets))"
  export CUDA_VISIBLE_DEVICES=$gpuId
  echo r=$MPI_LOCAL_RANK g=$CUDA_VISIBLE_DEVICES ib=$OMPI_MCA_btl_openib_if_include gpu=$gpuId ompthreads="$OMP_NUM_THREADS" numactl --cpunodebind="$cpuSocket" --preferred="$cpuSocket" $*
  #numactl --cpunodebind="$cpuSocket" --preferred="$cpuSocket" 
  $*
else
  echo "error no numactl"
  $*
fi
 exit $?

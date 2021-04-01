#!/usr/bin/env bash
# Copyright 2019-2021 Axel Huebl, Rene Widera, David Rogers
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


# PIConGPU batch script for Frontier's slurm (sbatch) batch system
#   https://www.olcf.ornl.gov/for-users/system-user-guides/frontier/frontier-user-guide/#running-jobs

#SBATCH --partition=!TBG_queue
#SBATCH --time=!TBG_wallTime
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --ntasks=!TBG_tasks
#SBATCH --ntasks-per-node=!TBG_gpusPerNode
#SBATCH --mincpus=!TBG_mpiTasksPerNode
#SBATCH --cpus-per-task=!TBG_coresPerGPU
#SBATCH --mem=0
#SBATCH --gres=gpu:!TBG_gpusPerNode
#SBATCH --chdir=!TBG_dstPath
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
# do not overwrite existing stderr and stdout files
#SBATCH --open-mode=append
#SBATCH -o stdout
#SBATCH -e stderr

## calculations will be performed by tbg ##
.TBG_queue=${TBG_QUEUE:-"amdMI100"}

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_nameProject=${proj:-""}
.TBG_profile=${PIC_PROFILE:-"${PROJWORK}/!TBG_nameProject/${USER}/picongpu.profile"}

# number of available/hosted GPUs per node in the system
.TBG_numHostedGPUPerNode=4

# required GPUs per node for the current job
.TBG_gpusPerNode=`if [ $TBG_tasks -gt $TBG_numHostedGPUPerNode ] ; then echo $TBG_numHostedGPUPerNode; else echo $TBG_tasks; fi`

# host memory per gpu
.TBG_memPerGPU="$((62000 / $TBG_gpusPerNode))"
# host memory per node
.TBG_memPerNode="$((TBG_memPerGPU * TBG_gpusPerNode))"

# number of cores to block per GPU -
.TBG_coresPerGPU=2

# We only start 1 MPI task per GPU
.TBG_mpiTasksPerNode="$(( TBG_gpusPerNode * 1 ))"

# use ceil to caculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_gpusPerNode - 1 ) / TBG_gpusPerNode))"

## end calculations ##

echo 'Running program...'
echo !TBG_jobName

cd !TBG_dstPath

source !TBG_profile 2>/dev/null
if [ $? -ne 0 ] ; then
  echo "Error: PIConGPU environment profile under \"!TBG_profile\" not found!"
  exit 1
fi
#set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput

export OMP_NUM_THREADS=!TBG_coresPerGPU
#mpiexec -tag-output --display-map !TBG_dstPath/input/bin/picongpu --mpiDirect !TBG_author $programParams | tee output
srun -n !TBG_tasks -N !TBG_nodes --cpu-bind=map_ldom:1 --mem-bind=local --exclusive \
     !TBG_dstPath/input/bin/picongpu --mpiDirect !TBG_author !TBG_programParams | tee output

#jsrun --nrs !TBG_tasks --tasks_per_rs 1 --cpu_per_rs !TBG_coresPerGPU --gpu_per_rs 1 --latency_priority GPU-CPU --bind rs --smpiargs="-gpu" !TBG_dstPath/input/bin/picongpu --mpiDirect !TBG_author !TBG_programParams | tee output

#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2022 PIConGPU contributors
# Authors: Mika Soren Voss, Rene Widera
# License: GPLv3+
#

help()
{
  echo "Validate KHI output data."
  echo ""
  echo "The test is evaluating the magnetic field growth rate with the corresponding analytic solution."
  echo ""
  echo "Usage:"
  echo "    validate.sh [-d dataPath] [inputSetPath]"
  echo ""
  echo "  -d | --data dataPath                 - path to simulation output data"
  echo "                                         Default: inputPath/simOutput"
  echo "  -h | --help                          - show help"
  echo ""
  echo "  inputSetPath                         - path to the simulation input set"
  echo "                                         Default: current directory"
}

# options may be followed by
# - one colon to indicate they has a required argument
OPTS=`getopt -o d:h -l data:,help -- "$@"`
if [ $? != 0 ] ; then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

eval set -- "$OPTS"

# parser
while true ; do
    case "$1" in
        -d|--data)
            dataPath=$2
            shift
            ;;
        -h|--help)
            echo -e "$(help)"
            shift
            exit 0
            ;;
        --) shift; break;;
    esac
    shift
done


# the first parameter is the project path
if [ $# -eq 1 ] ; then
    inputSetPath="$1"
else
    inputSetPath="./"
fi

if [ -z "$dataPath" ] ; then
    dataPath=$inputSetPath/simOutput
fi

# test for growth rate
MAINTEST="$inputSetPath/lib/python/test/KHI_growthRate"

# check that no particles get lost
ret=0
awk 'NR==2 {beginCount=$2} NR>1 && NF!=0{if(beginCount!=$2){print("Error: Particle loss over time. Number of particles first step="beginCount", last step="$2);exit 1}}' e_macroParticlesCount.dat
ret="$((ret+$?))"
awk 'NR==2 {beginCount=$2} NR>1 && NF!=0{if(beginCount!=$2){print("Error: Particle loss over time. Number of particles first step="beginCount", last step="$2);exit 1}}' i_macroParticlesCount.dat
ret="$((ret+$?))"

# analyse magnetic field growth rate
python $MAINTEST/MainTest.py $inputSetPath/include/picongpu/param/ $dataPath/
ret="$((ret+$?))"
exit $ret

Thermal electron test for EmZ paper
=============================================

.. sectionauthor:: Rene Widera <r.widera (at) hzdr.de>

Summit
======


.. code-block:: bash

    pic-compile -j 3 -c "-DPIC_USE_openPMD=ON" ~/workspace/picongpu/share/picongpu/examples/PaperThermal/ paper_binaries_2022
    # run within an interactive session
    bsub -P $proj -W 2:00 -nnodes 1 -Is /bin/bash
    cd paper_binaries_2022/params/PaperThermal/
    for((n=0;n<10;++n)) ; do
        outFile="result_${n}.txt"
        rm $outFile
        for((i=0;i<6;++i)) ; do
          shape="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SHAPE=\([A-Z]*\).*/\1/g')"; \
          solver="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SOLVER=\([A-Za-z]*\).*/\1/g')"; \
          echo run: "# $solver $shape" | tee -a $outFile; \
          jsrun --nrs 1 --tasks_per_rs 1 --cpu_per_rs 6 --gpu_per_rs 1 --latency_priority GPU-CPU --bind rs --smpiargs="-gpu" cmakePreset_$i/bin/picongpu -d 1 1 1 -g 192 192 192 -s 100 -p 5 --periodic 1 1 1 --mpiDirect | tee -a $outFile;
        done
        grep -e "calc" -e "#" $outFile
    done
    pattern=$(grep "#" result_0.txt | tr " " ".")
    for i in $(echo $pattern) ; do echo $i; cat result_*.txt | grep -e "calc" -e "#" | grep -A1 "$i" | grep -v -e "#" -e "-" | awk 'BEGIN{sum=0.0; count=0}{sum+=$7;count++}END{printf("%f sec/step\n",sum/count/100)}' ; done


.. code-block:: bash

    cd paper_binaries_2022/params/PaperThermal/
    outFile="thermal.txt"
    rm $outFile
    for((i=0;i<6;++i)) ; do
      shape="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SHAPE=\([A-Z]*\).*/\1/g')"; \
      solver="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SOLVER=\([A-Za-z]*\).*/\1/g')"; \
      echo run: "# $solver $shape" | tee -a $outFile; \
      cd "cmakePreset_$i"; \
      runName=${i}_${solver}_${shape}; \
      tbg -f -s bsub -t etc/picongpu/summit-ornl/gpu_batch.tpl -c etc/picongpu/1.cfg ../runs/$runName | tee -a $outFile; \
      cd -
    done
    cd runs
    for i in $(ls -w 1) ; do plot_chargeConservation_overTimeOneSpecies.py $i/simOutput/openPMD/simData_%T.h5 --export $i/chargeConservingOverTime.png; done

Profiling Summit
----------------

.. code-block:: bash

    pic-compile -j 3 -c "-DPIC_USE_openPMD=OFF -DALPAKA_CUDA_SHOW_CODELINES=ON -DCMAKE_CXX_FLAGS=-g -DALPAKA_CUDA_KEEP_FILES=ON" ~/workspace/picongpu/share/picongpu/examples/PaperThermal/ profilingEsirkepovTenaryNoLoopIf
    # run within an interactive session
    bsub -P $proj -W 2:00 -nnodes 1 -Is /bin/bash
    cd profilingEsirkepovTenaryNoLoopIf/params/PaperThermal/
    for((n=0;n<1;++n)) ; do
        outFile="result_${n}.txt"
        rm $outFile
        for((i=0;i<6;++i)) ; do
          shape="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SHAPE=\([A-Z]*\).*/\1/g')"; \
          solver="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SOLVER=\([A-Za-z]*\).*/\1/g')"; \
          echo run: "# $solver $shape" | tee -a $outFile; \
          jsrun --nrs 1 --tasks_per_rs 1 --cpu_per_rs 6 --gpu_per_rs 1 --latency_priority GPU-CPU --bind rs --smpiargs="-gpu" /sw/summit/nsight-compute/2021.1.0/ncu --import-source on --set full --kernel-id '::regex:KernelComputeCurrent:' -f -o pic_report_${n}_${solver}_${shape} cmakePreset_$i/bin/picongpu -d 1 1 1 -g 192 192 192 -s 5 -p 5 --periodic 1 1 1 --mpiDirect | tee -a $outFile;
        done
    done

Spock
=====

.. code-block:: bash

    pic-compile -j 3 -c "-DPIC_USE_openPMD=ON" ~/workspace/picongpu/share/picongpu/examples/PaperThermal/ amd_paper_binaries_2022
    # run within an interactive session
    salloc --time=2:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:1 --gpu-bind=closest --gpus-per-task=1 --mem-per-gpu=64000 -p caar -A $proj bash
    cd amd_paper_binaries_2022/params/PaperThermal/
    for((n=0;n<10;++n)) ; do
        outFile="amd_result_${n}.txt"
        rm $outFile
        for((i=0;i<6;++i)) ; do
          shape="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SHAPE=\([A-Z]*\).*/\1/g')"; \
          solver="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SOLVER=\([A-Za-z]*\).*/\1/g')"; \
          echo run: "# $solver $shape" | tee -a $outFile; \
          srun -n 1 cmakePreset_$i/bin/picongpu -d 1 1 1 -g 192 192 192 -s 100 -p 5 --periodic 1 1 1 | tee -a $outFile;
        done
        grep -e "calc" -e "#" $outFile
    done
    pattern=$(grep "#" amd_result_0.txt | tr " " ".")
    for i in $(echo $pattern) ; do echo $i; cat amd_result_*.txt | grep -e "calc" -e "#" | grep -A1 "$i" | grep -v -e "#" -e "-" | awk 'BEGIN{sum=0.0; count=0}{sum+=$7;count++}END{printf("%f sec/step\n",sum/count/100)}' ; done

.. code-block:: bash

    cd amd_paper_binaries_2022/params/PaperThermal/
    outFile="amd_thermal.txt"
    rm $outFile
    for((i=0;i<6;++i)) ; do
      shape="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SHAPE=\([A-Z]*\).*/\1/g')"; \
      solver="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SOLVER=\([A-Za-z]*\).*/\1/g')"; \
      echo run: "# $solver $shape" | tee -a $outFile; \
      cd "cmakePreset_$i"; \
      runName=${i}_${solver}_${shape}; \
      tbg -f -s sbatch -t etc/picongpu/spock-ornl/caar.tpl -c etc/picongpu/1.cfg ../amd_runs/$runName | tee -a $outFile; \
      cd -
    done
    cd amd_runs
    for i in $(ls -w 1) ; do plot_chargeConservation_overTimeOneSpecies.py $i/simOutput/openPMD/simData_%T.h5 --export $i/chargeConservingOverTime.png; done

Spock CPU
=========

.. code-block:: bash
    salloc --time=3:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=64 --gres=gpu:1  --gpus-per-task=1 --mem-per-gpu=64000 -p caar -A $proj bash
    export CXX=CC
    export PIC_BACKEND=omp2b:native
    pic-compile -j 6 -c "-DPIC_USE_openPMD=OFF -DPIC_USE_ISAAC=OFF" ~/workspace/picongpu/share/picongpu/examples/PaperThermal/ cpu_paper_binaries_2022
    cd cpu_paper_binaries_2022/params/PaperThermal/
    for((n=0;n<10;++n)) ; do
        outFile="cpu_result_${n}.txt"
        rm $outFile
        for((i=0;i<6;++i)) ; do
          shape="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SHAPE=\([A-Z]*\).*/\1/g')"; \
          solver="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SOLVER=\([A-Za-z]*\).*/\1/g')"; \
          echo run: "# $solver $shape" | tee -a $outFile; \
          srun -n 1 --cpu-bind=no cmakePreset_$i/bin/picongpu -d 1 1 1 -g 192 192 192 -s 100 -p 5 --periodic 1 1 1 | tee -a $outFile;
        done
        grep -e "calc" -e "#" $outFile
    done
    pattern=$(grep "#" cpu_result_0.txt | tr " " ".")
    for i in $(echo $pattern) ; do echo $i; cat cpu_result_*.txt | grep -e "calc" -e "#" | grep -A1 "$i" | grep -v -e "#" -e "-" | awk 'BEGIN{sum=0.0; count=0}{sum+=$8;count++}END{printf("%f sec/step\n",sum/count/100)}' ; done

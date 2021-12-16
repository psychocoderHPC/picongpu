Thermal electron test for EmZ paper
=============================================

.. sectionauthor:: Rene Widera <r.widera (at) hzdr.de>

Summit
======


.. code-block:: bash

    pic-compile -j 3 -c "-DPIC_USE_openPMD=ON" ~/workspace/picongpu/share/picongpu/examples/PaperThermal/ paper_binaries
    cd paper_binaries/params/PaperThermal/
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


.. code-block:: bash

    cd paper_binaries/params/PaperThermal/
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


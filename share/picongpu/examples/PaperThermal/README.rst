Thermal electron test for EmZ paper
=============================================

.. sectionauthor:: Rene Widera <r.widera (at) hzdr.de>

.. code-block:: bash

    pic-compile -j 10 -c "-DPIC_USE_openPMD=OFF" ~/workspace/picongpu/share/picongpu/examples/PaperThermal/ paperMatrix
    cd paperMatrix/params/PaperThermal/
    outFile="result.txt"
    rm $outFile
    for((i=0;i<9;++i)) ; do
      shape="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SHAPE=\([A-Z]*\).*/\1/g')"; \
      solver="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SOLVER=\([A-Za-z]*\).*/\1/g')"; \
      echo run: "# $solver $shape" | tee -a $outFile; \
      cmakePreset_$i/bin/picongpu -d 1 1 1 -g 256 256 256 -s 100 -p 5 --periodic 1 1 1 | tee -a $outFile;
    done
    grep -e "calc" -e "#" $outFile

.. code-block:: bash

    pic-compile -j 10 -c "-DPIC_USE_openPMD=OFF" ~/workspace/picongpu/share/picongpu/examples/PaperThermal/ paperMatrix
    cd paperMatrix/params/PaperThermal/
    outFile="result.txt"
    rm $outFile
    for((i=0;i<9;++i)) ; do
      shape="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SHAPE=\([A-Z]*\).*/\1/g')"; \
      solver="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*SOLVER=\([A-Za-z]*\).*/\1/g')"; \
      echo run: "# $solver $shape" | tee -a $outFile; \
      cd "cmakePreset_$i"; \
      tbg -f -s bash -t etc/picongpu/bash/mpiexec.tpl -c etc/picongpu/1.cfg runs/001_data | tee -a $outFile; \
      cd -
    done
    grep -e "calc" -e "#" $outFile

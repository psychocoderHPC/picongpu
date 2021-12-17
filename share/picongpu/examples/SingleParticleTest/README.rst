Single particle test for EmZ paper
=============================================

.. sectionauthor:: Rene Widera <r.widera (at) hzdr.de>


Single Particle Test
====================

.. code-block:: bash


outFile="result.txt"
rm $outFile
for((i=0;i<9;++i)) ; do
  shape="CIC"; \
  solver="$(cmakePreset_$i/cmakeFlags $i | sed 's/.*CURRENTSOLVER=\([A-Za-z]*\).*/\1/g')"; \
  echo run: "# $solver $shape" | tee -a $outFile; \
  runName=${i}_${solver}_${shape}; \
  cd "cmakePreset_$i"; \
  tbg -f -s bash -t ~/workspace/picongpu/etc/picongpu/bash/mpiexec.tpl -c etc/picongpu/1.cfg ../runs/$runName | tee -a $outFile; \
  cd -
done

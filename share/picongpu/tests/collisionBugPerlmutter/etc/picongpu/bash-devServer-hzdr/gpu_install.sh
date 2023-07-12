spack install --reuse cmake@3.24.2 %gcc@10.2.0
spack load cmake@3.24.2 %gcc@10.2.0

echo "openpmd-api:"
spack install --reuse openpmd-api@0.14.5 %gcc@10.2.0 \
    ^adios2@2.8.3 +cuda cuda_arch=70\
    ^cmake@3.24.2 \
    ^hdf5@1.12.2 \
    ^openmpi@4.1.3 +atomics +cuda cuda_arch=70\
    ^python@3.10.4 \
    ^py-numpy@1.23.3

echo "boost:"
spack install --reuse boost@1.78.0 \
    +program_options \
    +filesystem \
    +system \
    +math \
    +serialization \
    +fiber \
    +context \
    +thread \
    +chrono \
    +atomic \
    +date_time \
    ~python \
    %gcc@10.2.0

echo "pngwriter"
spack install --reuse pngwriter@0.7.0 %gcc@10.2.0

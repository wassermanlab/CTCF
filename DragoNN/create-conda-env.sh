#!/usr/bin/env bash

# i.e. enable conda (de)activate
eval "$(conda shell.bash hook)"

# Create DragoNN environment
conda create -n dragonn -c bioconda -c conda-forge deeptools=3.5 \
    h5py=2.10 matplotlib=3.3.2 mkl-service=2.3.0 numpy=1.19.1 \
    pybedtools=0.8.1 python=3.8.5 scikit-learn=0.23.2 yaml=0.2.5 \
    zlib=1.2.11

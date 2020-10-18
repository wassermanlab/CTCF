#!/usr/bin/env bash

# i.e. enable conda (de)activate
eval "$(conda shell.bash hook)"

# Create conda environment
conda create -n CTCF -c bioconda -c conda-forge -c pytorch biopython=1.78 \
    click=7.1.2 curl=7.71.1 jupyterlab=2.2.8 matplotlib=3.2.2 numpy=1.19.1 \
    pandas=1.1.3 plotly=4.11.0 python=3.8.5 seaborn=0.11.0 scikit-learn=0.23.2 \
    scipy=1.5.2 torchvision=0.7.0 tqdm=4.50.2

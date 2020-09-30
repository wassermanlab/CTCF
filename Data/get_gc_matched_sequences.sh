#!/bin/bash

# Download TF binding matrix
if [ ! -f matrix2d.ReMap+UniBind.sparse.npz ]; then
    wget http://expdata.cmmt.ubc.ca/downloads/TF-Binding-Matrix/matrix/UCSC/200bp/matrix2d.ReMap+UniBind.sparse.npz
fi
if [ ! -f regions_idx.pickle.gz ]; then
    wget http://expdata.cmmt.ubc.ca/downloads/TF-Binding-Matrix/matrix/UCSC/200bp/regions_idx.pickle.gz
fi
if [ ! -f sequences.200bp.fa ]; then
    wget http://expdata.cmmt.ubc.ca/downloads/TF-Binding-Matrix/matrix/UCSC/200bp/sequences.200bp.fa.gz
    gunzip sequences.200bp.fa.gz
fi
if [ ! -f tfs_idx.pickle.gz ]; then
    wget http://expdata.cmmt.ubc.ca/downloads/TF-Binding-Matrix/matrix/UCSC/200bp/tfs_idx.pickle.gz
fi

# Get a set of positive and negative CTCF-bound sequences with matched %GC
python get_gc_matched_sequences.py --matrix matrix2d.ReMap+UniBind.sparse.npz --seq-file sequences.200bp.fa --regions-idx regions_idx.pickle.gz --tf CTCF --tfs-idx tfs_idx.pickle.gz

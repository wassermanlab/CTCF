#!/bin/bash

# Download TF binding matrix
wget http://expdata.cmmt.ubc.ca/downloads/TF-Binding-Matrix/matrix/UCSC/200bp/matrix2d.ReMap+UniBind.sparse.npz
wget http://expdata.cmmt.ubc.ca/downloads/TF-Binding-Matrix/matrix/UCSC/200bp/regions_idx.pickle.gz
wget http://expdata.cmmt.ubc.ca/downloads/TF-Binding-Matrix/matrix/UCSC/200bp/tfs_idx.pickle.gz
wget http://expdata.cmmt.ubc.ca/downloads/TF-Binding-Matrix/matrix/UCSC/200bp/sequences.200bp.fa.gz

# Get a set of positive and negative CTCF-bound sequences with matched %GC
python get_gc_matched_sequences.py --matrix matrix2d.ReMap+UniBind.sparse.npz --seqfile sequences.200bp.fa --regions-idx regions_idx.pickle.gz --tf CTCF --tfs-idx tfs_idx.pickle.gz

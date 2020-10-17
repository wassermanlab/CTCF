#!/bin/bash

if [ ! -f matrix2d.ReMap+UniBind.sparse.npz ]; then
    wget http://expdata.cmmt.ubc.ca/downloads/TF-Binding-Matrix/matrix/UCSC/200bp/matrix2d.ReMap+UniBind.sparse.npz
fi
if [ ! -f regions_idx.pickle.gz ]; then
    wget http://expdata.cmmt.ubc.ca/downloads/TF-Binding-Matrix/matrix/UCSC/200bp/regions_idx.pickle.gz
fi
if [ ! -f sequences.200bp.fa.gz ]; then
    wget http://expdata.cmmt.ubc.ca/downloads/TF-Binding-Matrix/matrix/UCSC/200bp/sequences.200bp.fa.gz
fi
if [ ! -f tfs_idx.pickle.gz ]; then
    wget http://expdata.cmmt.ubc.ca/downloads/TF-Binding-Matrix/matrix/UCSC/200bp/tfs_idx.pickle.gz
fi

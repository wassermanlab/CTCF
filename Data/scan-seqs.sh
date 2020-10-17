#!/bin/bash

# Get CTCF predictions in the set of positive/negative CTCF-bound sequences
gunzip ./pos_seqs.fa.gz
python ../JASPAR-UCSC-tracks/scan_sequence.py --fasta-file ./pos_seqs.fa --profiles-dir ../JASPAR-UCSC-tracks/profiles/ --output-dir ./pos_seqs/
gzip ./pos_seqs.fa
gunzip ./neg_seqs.fa.gz
python ../JASPAR-UCSC-tracks/scan_sequence.py --fasta-file ./neg_seqs.fa --profiles-dir ../JASPAR-UCSC-tracks/profiles/ --output-dir ./neg_seqs/
gzip ./neg_seqs.fa
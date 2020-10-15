#!/bin/bash

# Get CTCF predictions in the set of positive CTCF-bound sequences
python ../JASPAR-UCSC-tracks/scan_sequence.py --fasta-file ./pos_seqs.fa --profiles-dir ../JASPAR-UCSC-tracks/profiles --threads 4 --profile MA0139.1
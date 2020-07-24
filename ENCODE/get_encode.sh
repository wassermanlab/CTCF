#!/bin/bash

python get_encode.py --genome mm10 --feature accessibility
python get_encode.py --genome mm10 --feature tf

# Extract uniform DNase-seq regions of length 200 bp
if [ ! -f ./DNase-seq.200bp.bed ]; then
    cut -f 1,2,4,6 ./DNase-seq.bed | awk '{print($1"\t"$2+$4-100"\t"$2+$4+100"\t"$3);}' | \
    LC_ALL=C sort --parallel=8 -T ./ -k1,1 -k2,2n > ./DNase-seq.200bp.bed
fi

# Extract uniform CTCF ChIP-seq regions of length 200 bp centered at the peak max
if [ ! -f ./ChIP-seq.CTCF.200bp.bed ]; then
    cut -f 1,2,4,6 ./ChIP-seq.CTCF.bed | awk '{print($1"\t"$2+$4-100"\t"$2+$4+100"\t"$3);}' | \
    LC_ALL=C sort --parallel=8 -T ./ -k1,1 -k2,2n > ./ChIP-seq.CTCF.200bp.bed
fi


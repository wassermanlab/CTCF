#!/bin/bash

# Initialize
DANQ_DIR=../DanQ
CHRX_DIR=../Genomes/X

# Make predictions with CTCF models trained w/ DragoNN
# https://github.com/kundajelab/dragonn

for M in CTCF CTCF-fwd
do

    # Initialize
    STAT_DICT=${DANQ_DIR}/${M}/model.pth.tar

    # mm10
    OUT_DIR=./mm10/${M}
    mkdir -p $OUT_DIR

    for T in cerebellum forebrain heart hindbrain intestine kidney liver lung midbrain stomach thymus
    do
        for S in pos_seqs neg_seqs
        do
            echo "*** predict ${T}, ${S}, ${M}"
            SEQ_FILE=../Sequences/mm10/${S}.${T}.fa
            OUT_FILE=${OUT_DIR}/${S}.${T}.txt.gz
            if [ ! -f $OUT_FILE ]; then
                python predict.py -f $SEQ_FILE -o $OUT_FILE -s $STAT_DICT -r
            fi
        done
    done

    for G in bosTau6 CHIR_1.0 equCab3 gorGor4 gorGor5 hg38 mm9 oviAri3 panPan3 panTro6 ponAbe3 susScr11
    do
        echo "*** predict ${G}, ${M}"
        OUT_DIR=./${G}
        mkdir -p $OUT_DIR
        SEQ_FILE=${CHRX_DIR}/${G}.chrX.fa
        OUT_FILE=./${OUT_DIR}/${M}.chrX.txt.gz
        if [ ! -f $OUT_FILE ]; then
            python predict-X.py -f $SEQ_FILE -o $OUT_FILE -s $STAT_DICT -r
        fi
    done



done

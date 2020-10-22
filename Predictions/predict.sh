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

    for G in bosTau6 CHIR_1.0 equCab3 gorGor5 hg38 mm9 oviAri3 panPan3 panTro6 ponAbe3 susScr11
    do
        echo "*** predict ${G}, ${M}"
        SEQ_FILE=${CHRX_DIR}/${G}.chrX.fa
        OUT_FILE=./${G}.chrX.txt.gz
        if [ ! -f $OUT_FILE ]; then
            python predict-X.py -f $SEQ_FILE -o $OUT_FILE -s $STAT_DICT -r
        fi
    done



done




for M in CTCF CTCF-fwd
do

    # Initialize
    STAT_DICT=${DANQ_DIR}/${M}/model.pth.tar

    # mm10
    OUT_DIR=./mm10/${M}
    mkdir -p $OUT_DIR

    for F in FILES
    do
        echo "*** predict ${T}, ${S}, ${M}"
        SEQ_FILE=../Sequences/mm10/${S}.${T}.fa
        OUT_FILE=${OUT_DIR}/${S}.${T}.txt.gz
        if [ ! -f $OUT_FILE ]; then
            python predict.py -f $SEQ_FILE -o $OUT_FILE -s $STAT_DICT -r
        fi
    done
done

# # Brad files
# ls ../DiscordantDomains/ | cut -d "." -f 1 | sort | uniq | perl -e \
# '
#     @sizes=("large", "small");
#     $model = "../DragoNN/CTCF.arch.json";
#     $weights = "../DragoNN/CTCF.weights.h5";
#     while ($genome = <>) {
#         chomp $genome;
#         mkdir $genome;
#         foreach $size (@sizes) {
#             $bed_file = "../DiscordantDomains/$genome.$size.discordantDomains.bed";
#             open($fh, "<:encoding(UTF-8)", $bed_file)
#                 or die "Could not open file \"$bed_file\": $!";
#             while ($region = <$fh>) {
#                 chomp $region;
#                 system "python ./predict.py --region $region --genome $genome --model $model --weights $weights -o $genome";
#             }
#             close($fh);
#         } 
#     }
# '

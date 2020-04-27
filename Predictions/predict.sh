#!/bin/bash

# Initialize
ARCH_FILE=../DragoNN/CTCF.arch.json
WEIGHTS_FILE=../DragoNN/CTCF.weights.h5

# Make DragoNN predictions with CTCF model
# https://github.com/kundajelab/dragonn

# mm10
OUT_DIR=./mm10
mkdir -p ${OUT_DIR}

for T in cerebellum forebrain heart hindbrain intestine kidney liver lung midbrain stomach thymus
do
    for S in pos_seqs neg_seqs
    do
        SEQ_FILE=../Sequences/mm10/${S}.${T}.fa
        OUT_FILE=${OUT_DIR}/${S}.${T}.txt
        # Extract uniform DNase-seq regions of length 150 bp
        if [ ! -f ${OUT_FILE} ]; then
           dragonn predict --sequences ${SEQ_FILE} --arch-file ${ARCH_FILE} --weights-file ${WEIGHTS_FILE} --output-file ${OUT_FILE}
        fi
    done
done

# Brad files
ls ../DiscordantDomains/ | cut -d "." -f 1 | sort | uniq | perl -e \
'
    @sizes=("large", "small");
    $model = "../DragoNN/CTCF.arch.json";
    $weights = "../DragoNN/CTCF.weights.h5";
    while ($genome = <>) {
        chomp $genome;
        mkdir $genome;
        foreach $size (@sizes) {
            $bed_file = "../DiscordantDomains/$genome.$size.discordantDomains.bed";
            open($fh, "<:encoding(UTF-8)", $bed_file)
                or die "Could not open file \"$bed_file\": $!";
            while ($region = <$fh>) {
                chomp $region;
                ($c, $s, $e) = split(/\t/, $region);
                system "python ./predict.py --region $c $s+1 $e --genome $genome --model $model --weights $weights -o $genome";
            }
            close($fh);
        } 
    }
'

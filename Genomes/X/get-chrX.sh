#!/bin/bash

for g in bosTau6 equCab3 gorGor4 hg38 mm9 oviAri3 panPan3 panTro6 ponAbe3 susScr11
do
    if [ ! -f ${g}.chrX.fa ]; then
        mkdir ${g}
        cd ${g}
        wget ftp://hgdownload.soe.ucsc.edu/goldenPath/${g}/bigZips/${g}.fa.gz
        gunzip ${g}.fa.gz
        csplit -s -z ${g}.fa '/>/' '{*}'
        for i in xx* ; do \
            n=$(sed 's/>// ; s/ .*// ; 1q' "$i") ; \
            mv "$i" "$n.fa" ; \
        done
        cd ..
        mv ${g}/chrX.fa ${g}.chrX.fa
        rm -rf ${g}
    fi
done

if [ ! -f gorGor5.chrX.fa ]; then
    efetch -db nuccore -id LT578347.1 -format fasta > gorGor5.chrX.fa
fi

if [ ! -f CHIR_1.0.chrX.fa ]; then
    efetch -db nuccore -id CM001739.1 -format fasta > CHIR_1.0.chrX.fa
fi

#!/bin/bash

for g in bosTau6 equCab3 gorGor4 hg38 mm10 oviAri3 panPan2 panTro6 ponAbe3
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
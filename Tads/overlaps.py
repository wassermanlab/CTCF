#!/usr/bin/env python

import os
from pybedtools import BedTool
import sys

a = BedTool("tss1.bed")
b = BedTool("tss2.bed")

tads = []

hic_dir = "/arc/project/ex-ofornes-1/Hi-C/"

for d in os.listdir(hic_dir):
    if os.path.isdir(os.path.join(hic_dir, d)):
        tads_file = os.path.join(hic_dir, d, "tads", "tads-hg38.50kb.bed")
        if os.path.exists(tads_file):
            c = BedTool(tads_file)
            for interval in c.intersect(a, wa=True):
                tads.append(("tss1", d, interval["chrom"], interval["start"], interval["end"], interval["name"]))
            for interval in c.intersect(b, wa=True):
                tads.append(("tss2", d, interval["chrom"], interval["start"], interval["end"], interval["name"]))

#hic_dir = "/home/ofornes/CTCF/Tads/hg38/"

#for f in os.listdir(hic_dir):
#    tads_file = os.path.join(hic_dir, f)
#    if os.path.exists(tads_file):
#        sys.stderr.write(tads_file)
#        c = BedTool(tads_file)
#        for interval in c.intersect(a, wa=True):
#            tads.append(("tss1", f, interval["chrom"], interval["start"], interval["end"], interval["name"]))
#        for interval in c.intersect(b, wa=True):
#            tads.append(("tss2", f, interval["chrom"], interval["start"], interval["end"], interval["name"]))

for tad in tads:
    print("\t".join(map(str, tad)))

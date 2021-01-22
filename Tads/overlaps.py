#!/usr/bin/env python

import os
from pybedtools import BedTool
import sys

# Initialize
intersections = []
tss = BedTool("Tss.bed")

# Untar and uncompress Hi-C.tgz (i.e. TADs from Hicarus, unpublished)
hic_dir = "./data/Hicarus/"

for d in os.listdir(hic_dir):
    if os.path.isdir(os.path.join(hic_dir, d)):
        tads_file = os.path.join(hic_dir, d, "tads", "tads-hg38.50kb.bed")
        if os.path.exists(tads_file):
            tads = BedTool(tads_file)
            for i in tads.intersect(tss, wa=True, wb=True):
                intersections.append((i.fields[-1], d, i.fields[0], i.fields[1],
                    i.fields[2], i.fields[3]))

# Unzip hg38.TADs.zip (i.e. TADs from the 3D Genome Browser)
hic_dir = "./data/3dGenomeBrowser/"

for f in os.listdir(hic_dir):
    tads_file = os.path.join(hic_dir, f)
    if os.path.exists(tads_file):
        tads = BedTool(tads_file)
        for i in tads.intersect(tss, wa=True, wb=True):
            intersections.append((i.fields[-1], f, i.fields[0], i.fields[1],
                i.fields[2], i.fields[3]))

for i in intersections:
    print("\t".join(map(str, i)))

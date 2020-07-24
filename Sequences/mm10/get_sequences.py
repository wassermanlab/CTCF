#!/usr/bin/env python

import argparse
import os
from pybedtools import BedTool

from __init__ import ENCODE, ParseUtils

usage_msg = """
usage: %s --encode-dir DIR --fasta-file FILE [-h] [options]
""" % os.path.basename(__file__)

help_msg = """%s
builds multiple matrix of CTCF-bound and open regions across
and cells/tissues from ENCODE

  --encode-dir DIR    output directory from get_encode.py
  --fasta-file FILE   from get_mm10.sh (i.e. mm10.fa)

optional arguments:
  -h, --help          show this help message and exit
  -o, --out-dir DIR   output directory (default = "./")
""" % usage_msg

#-------------#
# Functions   #
#-------------#

def parse_args():
    """
    This function parses arguments provided via the command line and returns an
    {argparse} object.
    """

    parser = argparse.ArgumentParser(add_help=False)

    # Mandatory args
    parser.add_argument("--encode-dir")
    parser.add_argument("--fasta-file")

    # Optional args
    optional_group = parser.add_argument_group("optional arguments")
    optional_group.add_argument("-h", "--help", action="store_true")
    optional_group.add_argument("-o", "--out-dir", default=".")

    args = parser.parse_args()

    check_args(args)

    return(args)

def check_args(args):
    """
    This function checks an {argparse} object.
    """

    # Print help
    if args.help:
        print(help_msg)
        exit(0)

    # Check mandatory arguments
    if not args.encode_dir or not args.fasta_file:
        error = ["%s\n%s" % (usage_msg, os.path.basename(__file__)), "error",
            "arguments \"--encode-dir\" \"--fasta-file\" are required\n"]
        print(": ".join(error))
        exit(0)

def main():

    # Parse arguments
    args = parse_args()

    # Build matrices
    build_matrix(args.encode_dir, args.fasta_file, args.out_dir)

def build_matrix(encode_dir, fasta_file, out_dir="."):
    """
    e.g. ./get_sequences.py --encode-dir ../ENCODE/ --fasta-file ../Genomes/mm10/mm10.fa
    """

    # Create output dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)   

    #######################################################
    # First of all, figure out the common samples between #
    # DNase-seq and CTCF ChIP-seq experiments.            #
    #######################################################

    pkl_file = "metadata.mm10.accessibility.tsv.pickle.gz"
    encodes_acc = ParseUtils.load_pickle(os.path.join(encode_dir, pkl_file))
    samples_acc = set([e.biosample_name for e in encodes_acc.values()])
    pkl_file = "metadata.mm10.tf.tsv.pickle.gz"
    encodes_tfs = ParseUtils.load_pickle(os.path.join(encode_dir, pkl_file))
    samples_tfs = set([e.biosample_name for e in encodes_tfs.values()])
    samples = samples_acc.intersection(samples_tfs)

    #######################################################
    # Now, for each sample, create a high-quality set of  #
    # positive and negative sequences for DragoNN:        #
    # (*) Positive sequences are CTCF ChIP-seq regions in #
    #     which the peak max overlaps an open region      #
    # (*) Negative sequences are open regions that do not #
    #     overlap a CTCF ChIP-seq region                  #
    #######################################################

    for sample in sorted(samples):

        # Skip if DNase-seq file for this sample already exist
        dnase_seq_file = os.path.join(out_dir, ".DNase-seq.%s.bed" % sample)
        if not os.path.exists(dnase_seq_file):

            # Initialize
            intervals = []
            a = BedTool(os.path.join(encode_dir, "DNase-seq.200bp.bed"))

            for interval in a:
                encode_acc = encodes_acc[interval.fields[3]]
                if encode_acc.biosample_name != sample:
                    continue
                intervals.append(interval)

            b = BedTool("\n".join(map(str, intervals)), from_string=True).saveas(dnase_seq_file)

        # Skip if DNase-seq file for this sample already exist
        chip_seq_file = os.path.join(out_dir, ".ChIP-seq.CTCF.%s.bed" % sample)
        if not os.path.exists(chip_seq_file):

            # Initialize
            intervals = []
            a = BedTool(os.path.join(encode_dir, "ChIP-seq.CTCF.200bp.bed"))

            for interval in a:
                encode_tfs = encodes_tfs[interval.fields[3]]
                if encode_tfs.biosample_name != sample:
                    continue
                intervals.append(interval)

            b = BedTool("\n".join(map(str, intervals)), from_string=True).saveas(chip_seq_file)

        # Skip if positive sequences for this sample already exist
        sequences_file = os.path.join(out_dir, "pos_seqs.%s.fa" % sample)
        if not os.path.exists(sequences_file):

            # Skip if positive regions for this sample already exist
            bed_file = os.path.join(out_dir, ".pos_seqs.%s.bed" % sample)
            if not os.path.exists(bed_file):

                # Initialize
                intervals = set()
                a = BedTool(chip_seq_file)
                b = BedTool(dnase_seq_file)

                for interval in a.intersect(b, sorted=True, wa=True, f=0.5, r=True, stream=True):
                    intervals.add(interval)

                c = BedTool("\n".join(map(str, intervals)), from_string=True).saveas(bed_file)

            # Get BED and FASTA files
            a = BedTool(bed_file)
            s =  a.sequence(fi=fasta_file)

            # Write
            with open(sequences_file, "w") as f:
                f.write(open(s.seqfn).read())

        # Skip if negative sequences for this sample already exist
        sequences_file = os.path.join(out_dir, "neg_seqs.%s.fa" % sample)
        if not os.path.exists(sequences_file):
        
            # Skip if positive regions for this sample already exist
            bed_file = os.path.join(out_dir, ".neg_seqs.%s.bed" % sample)
            if not os.path.exists(bed_file):

                # Initialize
                intervals = set()
                b = BedTool(chip_seq_file)
                a = BedTool(dnase_seq_file)

                for interval in a.subtract(b, sorted=True, A=True, stream=True):
                    intervals.add(interval)

                c = BedTool("\n".join(map(str, intervals)), from_string=True).saveas(bed_file)

            # Get BED and FASTA files
            a = BedTool(bed_file)
            s =  a.sequence(fi=fasta_file)

            # Write
            with open(sequences_file, "w") as f:
                f.write(open(s.seqfn).read())

#-------------#
# Main        #
#-------------#

if __name__ == "__main__":
    main()

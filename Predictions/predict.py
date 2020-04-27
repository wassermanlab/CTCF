#!/usr/bin/env python

import argparse
from Bio.Alphabet.IUPAC import unambiguous_dna
from Bio.Seq import Seq
from lxml import etree
import math
import os
import subprocess as sp

usage_msg = """
usage: %s --region STR --genome STR --model-file FILE
                  --weights-file FILE [-h] [-o]
""" % os.path.basename(__file__)

help_msg = """%s
some description

  --region STR        region (e.g. "chrX 73040486 73072588")
  --genome STR        genome assembly (e.g. "hg19")
  --model FILE        model (i.e. CTCF.arch.json)
  --weights FILE      model weights (i.e. CTCF.weights.h5) 

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
    parser.add_argument("--region", nargs=3)
    parser.add_argument("--genome")
    parser.add_argument("--model")
    parser.add_argument("--weights")

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
    if not args.region or not args.genome or not args.model or not args.weights:
        error = ["%s\n%s" % (usage_msg, os.path.basename(__file__)), "error",
            "arguments \"--region\" \"--genome\" \"--model\" \"--weights\" are required\n"]
        print(": ".join(error))
        exit(0)

def get_DNA_sequence_from_UCSC(chrom, start, end, genome):
    """
    Retrieve DNA sequence for given genomic region from UCSC.
    Note: the function assumes that the start position is 1-based.
    """

    # Initialize
    sequence = ""
    url = "http://genome.ucsc.edu/cgi-bin/das/%s/dna?segment=%s:%s,%s" % (
        genome, chrom, start, end)

    # Get XML
    xml = etree.parse(url, parser=etree.XMLParser())
    # Get sequence
    sequence = xml.xpath("SEQUENCE/DNA/text()")[0].replace("\n", "")

    return Seq(sequence, unambiguous_dna)

def main():

    # Parse arguments
    args = parse_args()

    # Make CTCF predictions
    make_predictions(args.region, args.genome, args.model,
        args.weights, args.out_dir)

def make_predictions(region, genome, model_file, weights_file,
    out_dir="."):
    """
    e.g. ./predict.py --region chrX 73040486 73072588 --genome hg19 --model ../DragoNN/CTCF.arch.json --weights ../DragoNN/CTCF.weights.h5
    """

    # Initialize
    l = 200
    offset = 100

    # Create output dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Fix region sequence
    chrom = region[0]
    start = int(math.floor(int(region[1]) / float(l))) * l - l + 1
    end = int(math.ceil(int(region[2]) / float(l))) * l + l

    # Skip if FASTA file already exists
    fasta_file = os.path.join(out_dir, "%s.%s:%s-%s.fa" % \
        (genome, chrom, region[1], region[2]))
    if not os.path.exists(fasta_file):

        # Get sequence
        seq = get_DNA_sequence_from_UCSC(chrom, start, end, genome)

        with open(fasta_file, "w") as f:
            # For each 200bp window with 100bp offset...
            for w in range(start, end - offset, offset):
                # Get index
                ix = w - start
                # Get sub-sequence
                subseq = seq[ix:ix+200]
                # Write
                f.write(">%s:%s-%s\n%s\n" % (chrom, w, w+l-1, str(subseq)))

    # Skip if DragoNN predictions already exist
    out_file = os.path.join(out_dir, "%s.%s:%s-%s.txt" % \
        (genome, chrom, region[1], region[2]))
    if not os.path.exists(out_file):

        # Options
        opts = "--sequences %s --arch-file %s --weights-file %s --output-file %s" % \
            (fasta_file, model_file, weights_file, out_file)
        # Run DragoNN
        sp.call(["dragonn predict %s" % opts], shell=True)

#-------------#
# Main        #
#-------------#

if __name__ == "__main__":
    main()

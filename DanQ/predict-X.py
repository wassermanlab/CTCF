#!/usr/bin/env python

import click
import os

from predict import _predict
from utils.io import parse_fasta_file, write

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.option(
    "-f", "--fasta-file",
    help="FASTA file with sequences.",
    metavar="FILENAME",
    required=True
)
@click.option(
    "-o", "--out-file",
    help="Output file.",
    metavar="FILENAME",
    required=True
)
@click.option(
    "-r", "--rev-complement",
    help="Predict on reverse complement sequences.",
    is_flag=True,
    default=False
)
@click.option(
    "-s", "--state-dict",
    help="Model state dict to use.",
    metavar="FILENAME",
    required=True
)
@click.option(
    "-t", "--threads",
    default=1,
    help="Number of CPU threads to use.",
    show_default=True
)

def predictx(fasta_file, out_file, state_dict, rev_complement=False, threads=1):

    # Sequences
    fasta_file_200bp = "%s.200bp" % fasta_file
    if not os.path.exists(fasta_file_200bp):
        for seq_record in parse_fasta_file(fasta_file):
            chrom = seq_record.id
            seq = str(seq_record.seq)
            for i in range(0, len(seq) - 100, 100):
                write(
                    fasta_file_200bp,
                    ">%s:%s-%s\n%s" % (chrom, i+1, i+200, seq[i:i+200])
                )
                if i+200+100 > len(seq):
                    break

    _predict(fasta_file_200bp, out_file, state_dict, rev_complement, threads)

if __name__ == "__main__":
    predictx()
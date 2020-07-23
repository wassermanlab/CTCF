import os
import argparse
import pandas as pd
import gzip
import numpy as np
from Bio import SeqIO
from Bio.Seq import reverse_complement
from Bio.SeqUtils import GC

parser = argparse.ArgumentParser(description="Sample strand-specific GC comparable sequences for the TF (pos/neg sets)")
parser.add_argument("--scans-file", default=None, help="File with motif predictions from JASPAR")
parser.add_argument("--neg-file", default=None, help="FASTA file with unbound sequences (i.e. neg)")
parser.add_argument("--pos-file", default=None, help="FASTA file with bound sequences (i.e. pos)")
parser.add_argument("--out-dir", default=".", help="Directory where to output sampled sequences")

args = parser.parse_args()

scans_file = args.scans_file
neg_file = args.neg_file
pos_file = args.pos_file
out_dir = args.out_dir

##########################################################
# Load the fwd/rev motif predictions
##########################################################
nonzero = set()
fwd = set()
rev = set()

# Read in chunks
for chunk in pd.read_csv(scans_file, compression="gzip", header=None,
                             encoding="utf8", sep="\t", chunksize=1024,
                             comment="#", engine="python"):
    for index, row in chunk.iterrows():
        row = row.tolist()
        if row[-1] == "+":
            fwd.add(row[0])
        else:
            rev.add(row[0])

# Ignore ambiguous sequences with CTCF motifs in both strands
for i in list(fwd ^ rev):
    nonzero.add(i)

print("Data is loaded!")
##########################################################


##########################################################
# Extracting the fasta sequences for 0s and 1s
##########################################################
zero = set()
fasta_ids_nonzero = {}
fasta_ids_zero = {}

for fasta in SeqIO.parse(open(neg_file), "fasta"):
    name, sequence = fasta.id, str(fasta.seq)
    name = int(name.split(":")[0])
    fasta_ids_zero[name] = sequence
    fasta_ids_zero[-name] = reverse_complement(sequence)

for fasta in SeqIO.parse(open(pos_file), "fasta"):
    name, sequence = fasta.id, str(fasta.seq)
    name = int(name.split(":")[0])
    if name in nonzero:
        if name in fwd:
            fasta_ids_nonzero.setdefault("fwd", {})
            fasta_ids_nonzero.setdefault("rev", {}) 
            fasta_ids_nonzero["fwd"][name] = sequence
            fasta_ids_nonzero["rev"][-name] = reverse_complement(sequence)
        else:
            fasta_ids_nonzero.setdefault("fwd", {})
            fasta_ids_nonzero.setdefault("rev", {})
            fasta_ids_nonzero["rev"][name] = sequence
            fasta_ids_nonzero["fwd"][-name] = reverse_complement(sequence)

fasta_ids_zero = pd.Series(fasta_ids_zero)
print("Sequences are extracted!")
##########################################################

for strand in ["fwd", "rev"]:

    print("Strand is %s!" % strand)

    ######################################################
    # Sample new 0s
    ######################################################
    data_series = pd.Series(fasta_ids_nonzero[strand])
    nonzero_gc = data_series.apply(lambda x: GC(x.upper()))
    zero_gc = fasta_ids_zero.apply(lambda x: GC(x.upper()))

    bins = [0,10,20,30,40,50,60,70,80,90,100]
    labels = [10,20,30,40,50,60,70,80,90,100]

    #assigning bins from nonzero
    binned_nonzero = pd.cut(nonzero_gc, bins = bins, labels = labels)
    #assigning bins from zero
    binned_zero = pd.cut(zero_gc, bins = bins, labels = labels)

    #sampling new zeros
    new_zero_ind = []
    for l in labels:
        num_nonzero = len(binned_nonzero[binned_nonzero == l])
        num_zero = len(binned_zero[binned_zero == l])

        #if there are no nonzero peaks, continue
        if num_nonzero == 0 or num_zero == 0:
            continue

        if num_zero >= num_nonzero:
            #sample without replacement
            sampled_bins = binned_zero[binned_zero == l].sample(n=num_nonzero, replace=False)
            new_zero_ind = new_zero_ind + list(sampled_bins.index)

        if num_nonzero > num_zero:
            print("For bin %s we have more nonzeros than zeros!" % l)
            sampled_bins = binned_zero[binned_zero == l]
            new_zero_ind = new_zero_ind + list(sampled_bins.index)

    fasta_new_ids_zero = fasta_ids_zero[new_zero_ind]
    new_zero_gc = fasta_new_ids_zero.apply(lambda x: GC(x.upper()))
    print("Sequences are sampled!")
    ######################################################

    ######################################################
    # Saving files
    ######################################################
    fasta_file = os.path.join(out_dir, "neg_seqs.%s.fa" % strand)
    with open(fasta_file, "w") as f:
        for items in fasta_new_ids_zero.iteritems(): 
            name, sequence = items
            f.write(">" + str(name) + "\n")
            f.write(sequence + "\n")

    fasta_file = os.path.join(out_dir, "pos_seqs.%s.fa" % strand)
    with open(fasta_file, "w") as f:
        for items in data_series.iteritems():
            name, sequence = items
            f.write(">" + str(name) + "\n")
            f.write(sequence + "\n")

    print("Files are saved!")
    ######################################################

print("You are good to go!")
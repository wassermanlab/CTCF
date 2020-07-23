import os
import argparse
import pickle
import pandas as pd
import gzip
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils import GC

parser = argparse.ArgumentParser(description="Sample GC comparable sequences for the TF (pos/neg sets)")
parser.add_argument("--matrix", default= None, help="Path to the matrix with regions-TFs")
parser.add_argument("--seq-file", default=None, help="File with FASTA sequences")
parser.add_argument("--regions-idx", default=None, help="Idx file with regions")
parser.add_argument("--tfs-idx", default=None, help="Idx file with TFs")
parser.add_argument("--tf", default="CTCF", help="TF for which we need sample sequences")
parser.add_argument("--out-dir", default=".", help="Directory where to output sampled sequences")

args = parser.parse_args()

matrix = args.matrix
seq_file = args.seq_file
regions_idx = args.regions_idx
tfs_idx = args.tfs_idx
tf = args.tf
out_dir = args.out_dir

##########################################################
# Load the data set
##########################################################
#load the matrix
data = np.load(matrix)

for i in data.files:
    matrix2d = data[i] #(1817918, 163)
    
with gzip.open(regions_idx, 'rb') as f:
    regions = pickle.load(f) #1817918
    
with gzip.open(tfs_idx, 'rb') as f:
    tfs = pickle.load(f) #163
    
#make sure that everything is aligned
tfs = pd.Series(tfs).sort_values()
regions = pd.Series(regions).sort_values()

#extract the TF
tf_regions = pd.Series(matrix2d[:,tfs[tf]], index=regions.values)
tf_regions = tf_regions.dropna()

print("Data is loaded!")
##########################################################

##########################################################
# Extracting the fasta sequences for 0s and 1s
##########################################################
#none zero peaks
nonzero = tf_regions[tf_regions == 1].index
#zero peaks
zero = tf_regions[tf_regions == 0].index

fasta_ids_nonzero = {}
fasta_ids_zero = {}

fasta_sequences = SeqIO.parse(open(seq_file), "fasta")

tf_sequences = {}
for fasta in fasta_sequences:
    name, sequence = fasta.id, str(fasta.seq)
    
    name = name.split(":")[0]
    
    if int(name) in nonzero:
        #delete sequences with Ns
        if "N" not in sequence.upper():
            fasta_ids_nonzero[int(name)] = sequence.upper()
    elif int(name) in zero:
        #delete sequences with Ns
        if "N" not in sequence.upper():
            fasta_ids_zero[int(name)] = sequence.upper()

fasta_ids_nonzero = pd.Series(fasta_ids_nonzero)
fasta_ids_zero = pd.Series(fasta_ids_zero)
print("Sequences are extracted!")
##########################################################

##########################################################
# Sample new 0s
##########################################################
nonzero_gc = fasta_ids_nonzero.apply(lambda x: GC(x.upper()))
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
##########################################################

##########################################################
# Saving files
##########################################################
fasta_file = os.path.join(out_dir, "neg_seqs.fa")
with open(fasta_file, "w") as f:
    for items in fasta_new_ids_zero.iteritems(): 
        name, sequence = items
        f.write(">" + str(name) + "\n")
        f.write(sequence + "\n")

fasta_file = os.path.join(out_dir, "pos_seqs.fa")
with open(fasta_file, "w") as f:
    for items in fasta_ids_nonzero.iteritems(): 
        name, sequence = items
        f.write(">" + str(name) + "\n")
        f.write(sequence + "\n")

print("Files are saved! You are good to go!")
##########################################################

import numpy as np
import torch

from dragonn import one_hot_encode, reverse_complement

class Dataset(torch.utils.data.Dataset):
    """
    Adapted from:
    https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """

    def __init__(self, data, labels):
        """Initialization."""
        self.data = data
        self.labels = labels

    def __len__(self):
        """Total number of samples."""
        return(len(self.data))

    def __getitem__(self, idx):
        """Generates one sample of data."""
        return(self.data[idx], self.labels[idx])

def build_dataset(data, labels, indices, rev_complement=False):

    data = data[indices]
    labels = labels[indices]

    if rev_complement:
        data = np.append(data, reverse_complement(data), axis=0)
        labels = np.append(labels, labels, axis=0)

    return(Dataset(data, labels))

def split_data(pos_sequences, neg_sequences, seed=123):

    from copy import copy
    from sklearn.model_selection import train_test_split

    # One hot encode positive sequences
    encoded_seqs = one_hot_encode_fasta_file(pos_sequences)
    data = copy(encoded_seqs)
    labels = np.array([[1.]] * len(encoded_seqs))

    # One hot encode negative sequences
    encoded_seqs = one_hot_encode_fasta_file(pos_sequences)
    data = np.append(data, encoded_seqs, axis=0)
    labels = np.append(labels, np.array([[0.]] * len(encoded_seqs)), axis=0)

    # Split data
    indices = list(range(len(data)))
    train, test = train_test_split(indices, test_size=0.2, random_state=seed)
    validation, test = train_test_split(test, test_size=0.5, random_state=seed)
    splits = {"train": train, "validation": validation, "test": test}

    return(data, labels, splits)

def one_hot_encode_fasta_file(fasta_file):
    """One hot encodes sequences in a FASTA file."""

    from .io import parse_fasta_file

    # Initialize
    seqs = []

    for seq_record in parse_fasta_file(fasta_file):
        seqs.append(str(seq_record.seq).upper())

    return(one_hot_encode(seqs))
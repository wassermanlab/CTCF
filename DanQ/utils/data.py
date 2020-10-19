import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from .io import parse_fasta_file

# Defaults
default_parameters = dict(batch_size=64, shuffle=True, num_workers=1)

def get_data_loaders(tensor_datasets, kwargs=default_parameters):

    data_loaders = {}

    for k, v in tensor_datasets.items():
        data_loaders.setdefault(k, DataLoader(v, **kwargs))

    return(data_loaders)

def get_tensor_datasets(data_splits):

    tensor_datasets = {}

    for k, v in data_splits.items():
        data = torch.Tensor(v[0])
        labels = torch.Tensor(v[1])
        tensor_datasets.setdefault(k, TensorDataset(data, labels))

    return(tensor_datasets)

def split_data(pos_sequences, neg_sequences, rev_complement=False, seed=123):

    # Data
    pos_encoded_seqs = one_hot_encode_FASTA_file(pos_sequences)
    neg_encoded_seqs = one_hot_encode_FASTA_file(neg_sequences)
    data = np.concatenate((pos_encoded_seqs, neg_encoded_seqs))

    # Labels
    pos_labels = np.ones((len(pos_encoded_seqs), 1))
    neg_labels = np.zeros((len(neg_encoded_seqs) ,1))
    labels = np.concatenate((pos_labels, neg_labels))

    # Data splits
    indices = list(range(len(data)))
    train, test = train_test_split(indices, test_size=0.2, random_state=seed)
    validation, test = train_test_split(test, test_size=0.5, random_state=seed)
    data_splits = {
        "train": [data[train], labels[train]],
        "validation": [data[validation], labels[validation]],
        "test": [data[test], labels[test]]
    }

    # Reverse complement
    if rev_complement:
        data, labels = data_splits["train"][0], data_splits["train"][1]
        data_splits["train"][0] = np.append(
            data, reverse_complement(data), axis=0
        )
        data_splits["train"][1] = np.append(labels, labels, axis=0)

    return(data_splits)

def one_hot_encode_FASTA_file(fasta_file):
    """One hot encodes sequences in a FASTA file."""

    # Initialize
    encoded_seqs = []

    for seq_record in parse_fasta_file(fasta_file):
        encoded_seqs.append(one_hot_encode(str(seq_record.seq).upper()))

    return(np.array(encoded_seqs))

def one_hot_encode(seq):
    """One hot encodes a sequence."""

    seq = seq.replace("A", "0")
    seq = seq.replace("C", "1")
    seq = seq.replace("G", "2")
    seq = seq.replace("T", "3")

    encoded_seq = np.zeros((4, len(seq)), dtype="float16")

    for i in range(len(seq)):
        if seq[i].isdigit():
            encoded_seq[int(seq[i]), i] = 1
        else:
            # i.e. Ns
            encoded_seq[:, i] = 0.25

    return(encoded_seq)

def one_hot_decode(encoded_seq):
    """Reverts a sequence's one hot encoding."""

    seq = []
    code = list("ACGT")
 
    for i in encoded_seq.transpose(1, 0):
        try:
            seq.append(code[int(np.where(i == 1)[0])])
        except:
            # i.e. N?
            seq.append("N")

    return("".join(seq))

def reverse_complement(encoded_seqs):
    """Reverse complements a list of one hot encoded sequences."""
    return(encoded_seqs[..., ::-1, ::-1])
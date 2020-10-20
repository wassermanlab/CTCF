# from Bio import SeqIO
# import gzip
# import math
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import seaborn as sns
# from sklearn.metrics import (
#     average_precision_score, precision_recall_curve,
#     roc_auc_score, roc_curve,
#     matthews_corrcoef
# )
# from sklearn.model_selection import train_test_split
# from time import time
# import torch.nn as nn
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# from utils.pytorchtools import EarlyStopping

from Bio import SeqIO
import click
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.danq import DanQ
from utils.io import parse_fasta_file, write
from utils.data import one_hot_encode, reverse_complement
#from utils.predictor import Predictor

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

def predict(fasta_file, out_file, state_dict, rev_complement=False, threads=1):

    # Sequences
    sequences = []
    for seq_record in parse_fasta_file(fasta_file):
        sequences.append((seq_record.id, str(seq_record.seq).upper()))
    df = pd.DataFrame(sequences, columns=["Id", "Sequence"])

    # One-hot encode
    encoded_sequences = []
    for seq in df["Sequence"]:
        encoded_sequences.append(one_hot_encode(seq))
    encoded_sequences = np.array(encoded_sequences)

    # TensorDataset
    ix = np.array([[i] for i in range(len(sequences))])
    dataset = TensorDataset(torch.Tensor(encoded_sequences), torch.Tensor(ix))
    if rev_complement:
        encoded_sequences_rc = np.array(reverse_complement(encoded_sequences))
        dataset_rc = TensorDataset(
            torch.Tensor(encoded_sequences_rc), torch.Tensor(ix)
        )

    # DataLoader
    parameters = dict(batch_size=64, num_workers=threads)
    dataloader = DataLoader(dataset, **parameters)
    if rev_complement:
        dataloader_rc = DataLoader(dataset_rc, **parameters)

    # Predict
    sequence_length = len(sequences[0][1])
    predictions = __predict(sequence_length, 1, dataloader, state_dict)
    if rev_complement:
        predictions_rc = __predict(
            sequence_length, 1, dataloader_rc, state_dict
        )
    else:
        predictions_rc = np.empty((len(predictions)))
        predictions_rc[:] = np.NaN

    # Save predictions
    zipped_predictions = np.array(
        list(zip(df["Id"].to_list(), predictions, predictions_rc[:]))
    )
    df = pd.DataFrame(zipped_predictions, columns=["Id", "Fwd", "Rev"])
    df.to_csv(out_file, compression="gzip", index=False)

def __predict(sequence_length, n_features, dataloader, state_dict):

    predictions = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DanQ(sequence_length, n_features).to(device)
    model.load_state_dict(torch.load(state_dict))
    model.eval() # set the model in evaluation mode

    for seqs, labels in dataloader:
        x = seqs.to(device) # shape = (batch_size, 4, 200)
        labels = labels.to(device)
        with torch.no_grad():
            # Forward pass
            outputs = model(x)
            # Save predictions
            if predictions is None:
                predictions = outputs.data.cpu().numpy()
            else:
                predictions = np.append(
                    predictions, outputs.data.cpu().numpy(), axis=0
                )

    return(predictions.flatten())

if __name__ == "__main__":
    predict()
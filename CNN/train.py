import click
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

conv_kernel_size = 8
pool_kernel_size = 4

from selene import Trainer, initialize_model
from utils.io import write
from utils.data import build_dataset, split_data

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.option(
    "-a", "--architecture",
    help="Model architecture.",
    default="deeperdeepsea",
    metavar="STRING",
    show_default=True,
    type=click.Choice(
        ["danq"],
        case_sensitive=False
    )
)
@click.option(
    "-l", "--learn-rate",
    help="Learning rate.",
    default=0.003,
    show_default=True
)
@click.option(
    "-n", "--name",
    help="Transcription factor name.",
    required=True,
    type=str
)
@click.option(
    "--neg-sequences",
    help="FASTA file with negative sequences.",
    metavar="FILENAME",
    required=True,
)
@click.option(
    "-o", "--out-dir",
    default="./",
    help="Output directory.",
    metavar="DIRECTORY",
    show_default=True
)
@click.option(
    "--pos-sequences",
    help="FASTA file with positive sequences.",
    metavar="FILENAME",
    required=True,
)
@click.option(
    "-r", "--rev-complement",
    help="Train on reverse complement sequences.",
    is_flag=True,
    default=False
)
@click.option(
    "-s", "--seed",
    default=123,
    help="Seed for random generation.",
    show_default=True
)
@click.option(
    "-t", "--threads",
    default=1,
    help="Number of CPU threads to use.",
    show_default=True
)
@click.option(
    "-v", "--verbose",
    help="Verbose mode.",
    is_flag=True,
    default=False
)

def train(
    name, neg_sequences, pos_sequences, architecture="deeperdeepsea",
    learn_rate=0.003, out_dir="./", rev_complement=False, seed=123, threads=1,
    verbose=False
):
    """Train a model."""

    if verbose:
        write(None, "*** Loading sequence data...")

    # Splits
    data, labels, splits = split_data(pos_sequences, neg_sequences, seed)
    total_sequences, sequence_length, one_hot_encoding_size = data.shape
    print("data shape", data.shape)

    # Datasets
    datasets = {
        "train": build_dataset(data, labels, splits["train"], rev_complement),
        "validation": build_dataset(data, labels, splits["validation"]),
        "test": build_dataset(data, labels, splits["test"])
    }

    # Generators
    parameters = {"batch_size": 64, "shuffle": True, "num_workers": threads}
    generators = {
        "train": DataLoader(datasets["train"], **parameters),
        "validation": DataLoader(datasets["validation"], **parameters),
        "test": DataLoader(datasets["test"], **parameters)
    }

    inputs, targets = next(iter(generators["train"]))
    print("input shape", inputs.shape)

    if verbose:
        write(None, "*** Initializing model...")

    # Initialize model
    model, criterion, optimizer = initialize_model(
        architecture, sequence_length, learn_rate
    )

    if verbose:
        write(None, "*** Training/Validating model...")

    # Train/validate model
    trainer = Trainer(
        model, criterion, optimizer, dict([(0, name)]), generators,
        cpu_n_threads=threads, output_dir=out_dir, verbose=verbose
    )
    trainer.train_and_validate()
    exit(0)

    # model : torch.nn.Module
    #     The model to train.
    # criterion : torch.nn._Loss
    #     The loss function to optimize.
    # optimizer : torch.optim.Optimizer
    #     The optimizer to minimize loss with.
    # feature_index : dict
    #     A dictionary that maps feature indices (`int`) to names (`int`).
    # generators : dict
    #     A dictionary that maps the `train`, `validation` and `test` steps
    #     to `torch.utils.data.DataLoader` instances.


    if verbose:
        write(None, "*** Initializing model...")
    model = DeepSEA(X_train.shape[2], 1)



    # Initialize
    max_epochs = 100

    # if verbose:
    #     write(None, "*** Initializing PyTorch...")

    # # CUDA for PyTorch
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    # torch.backends.cudnn.benchmark = True

    # # Set manual seed for reproducibility
    # torch.manual_seed(seed)



    # train
    print("starting model training...")
    model.train(X_train, y_train, validation_data=(X_valid, y_valid))
    valid_result = model.test(X_valid, y_valid)
    print("final validation metrics:")
    print(valid_result)
    # save
    print("saving model files..")
    model.save(prefix)
    print("Done!")

if __name__ == "__main__":
    train()

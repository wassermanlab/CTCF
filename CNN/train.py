import click
import numpy as np

from utils.io import write
from utils.data import get_data_loaders, get_tensor_datasets, split_data
from utils.trainer import Trainer

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.option(
    "-a", "--architecture",
    help="Model architecture.",
    default="danq",
    metavar="STRING",
    show_default=True,
    type=click.Choice(["danq", "deepsea"], case_sensitive=False)
)
@click.option(
    "-l", "--learn-rate",
    help="Learning rate.",
    default=0.003,
    show_default=True
)
@click.option(
    "-m", "--max-epochs",
    help="Max. number of epochs.",
    default=100,
    show_default=True
)
@click.option(
    "-n", "--name",
    help="Feature name.",
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
    name, neg_sequences, pos_sequences, architecture="danq", learn_rate=0.003,
    max_epochs=100, out_dir="./", rev_complement=False, seed=123, threads=1,
    verbose=False
):
    """Train a model."""

    if verbose:
        write(None, "*** Loading data...")

    # Data splits
    data_splits = split_data(
        pos_sequences, neg_sequences, rev_complement, seed
    )

    # Tensor datasets
    tensor_datasets = get_tensor_datasets(data_splits)

    # Data loaders
    # parameters = dict(batch_size=64, shuffle=True, num_workers=threads)
    parameters = dict(batch_size=64)
    data_loaders = get_data_loaders(tensor_datasets, parameters)

    if verbose:
        write(None, "*** Training model...")

    # Train/validate model
    features = {0: name}
    trainer = Trainer(
        architecture, features, data_loaders, learn_rate, max_epochs,
        out_dir, verbose
    )
    trainer.test()
    trainer.compute_performance_metrics()

if __name__ == "__main__":
    train()
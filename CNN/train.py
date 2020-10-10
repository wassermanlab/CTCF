import click
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader


from models.danq import DanQ, get_criterion, get_optimizer
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
    default=0.01,
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

    if verbose:
        write(None, "*** Initializing model...")

    # Initialize model
    model, criterion, optimizer = __initialize_model(
        architecture, sequence_length, learn_rate
    )

    if verbose:
        write(None, "*** Training/Validating model...")

    # Use CUDA
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:0")
        model.cuda()
        criterion.cuda()
    else:
        device = torch.device("cpu")

    # Train/validate model
    for epoch in range(10):

        model.train() #tell model explicitly that we train
        running_loss = 0.0
        for inputs, targets in generators["train"]:

            inputs = inputs.float().to(device)
            targets = targets.float().to(device)

            #zero the existing gradients so they don't add up
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs.transpose(1, 2))
            loss = criterion(outputs, targets) 
            # Backward and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #save training loss 
        # return(running_loss / len(self.generators["train"]))
        print(running_loss / len(generators["train"]))

def __initialize_model(architecture, sequence_length, lr=0.01):
    """
    Adapted from:
    https://selene.flatironinstitute.org/utils.html#initialize-model

    Initialize model (and associated criterion, optimizer)

    Parameters
    ----------
    architecture : str
        Available model architectures: `danq`, `deeperdeepsea`, `deepsea` and
        `heartenn`.
    sequence_length : int
        Model-specific configuration
    lr : float
        Learning rate.

    Returns
    -------
    tuple(torch.nn.Module, torch.nn._Loss, torch.optim)
        * `torch.nn.Module` - the model architecture
        * `torch.nn._Loss` - the loss function associated with the model
        * `torch.optim` - the optimizer associated with the model
    """
    model = DanQ(sequence_length, 1)
    criterion = get_criterion()
    optimizer = get_optimizer(model.parameters(), lr)

    return(model, criterion, optimizer)

if __name__ == "__main__":
    train()

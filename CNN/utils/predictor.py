import numpy as np
import os
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
import torch

from utils.io import write

metrics = dict(
    roc_auc=roc_auc_score,
    average_precision=average_precision_score,
    m_corr_coef=matthews_corrcoef
)

class Predictor(object):
    """
    Adapted from:
    https://selene.flatironinstitute.org/selene.html#trainmodel

    This class ties together the various objects and methods needed to train
    and validate a model.

    {Predictor} saves a checkpoint model after every epoch as well as a
    best-performing model to `output_dir`.

    {Predictor} also outputs the training and validation losses to monitor if the
    model is still improving of if there are signs of overfitting, etc.

    Parameters
    ----------
    architecture : str
        The model architecture to train.
    state_dict : str
        Path to model's `state_dict`.
    generators : dict
        A dictionary that maps the `train`, `validation` and `test` steps
        to `torch.utils.data.DataLoader` instances.
    lr: float, optional
        Default is 0.003. Sets the learning rate.
    max_epochs : int, optional
        Default is 100. The maximum number of epochs to iterate over.
    output_dir : str, optional
        Default is current working directory. The output directory to save
        model checkpoints and logs in.
    threads : int, optional
        Default is 1. Sets the number threads for CPU operations.
    verbose: bool, optional
        Default is `False`.

    Attributes
    ----------
    criterion : `torch.nn._Loss`
        The loss function to optimize.
    features : dict
        The name of each feature.
    generator : `torch.utils.data.DataLoader`
        The generator for the `train`, `validation` and `test` sets.
    max_epochs : int
        The maximum number of epochs to iterate over.
    metrics : dict
        A dictionary that maps metric names (`str`) to metric functions. By
        default, this contains `"roc_auc"`, which maps to
        `sklearn.metrics.roc_auc_score`, `"average_precision"`, which maps to
        `sklearn.metrics.average_precision_score`, and `"m_corr_coef"`, which
        maps to `sklearn.metrics.matthews_corrcoef`.
    model : `torch.nn.Module`
        The model to train.

    optimizer : `torch.optim.Optimizer`
        The optimizer to minimize loss with.
    output_dir : str
        The directory to save model checkpoints and files.
    """

    def __init__(
        self, architecture, state_dict, generator, output_dir="./",
        verbose=False
    ):
        """
        Constructs a new `Predictor` object.
        """

        self.generator = generator
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._verbose = verbose

        # Load model
        model_file

    def __initialize_model(self, architecture, lr=0.003):
        """
        Adapted from:
        https://selene.flatironinstitute.org/utils.html#initialize-model

        Initialize model (and associated criterion and optimizer)

        Parameters
        ----------
        architecture : str
            Available model architectures: `danq`.
        lr : float
            Learning rate.
        """

        if architecture == "danq":
            from models.danq import DanQ

        self.model = model_class(data.shape[-1], 1).to(self._device)
        self.criterion = get_criterion()
        self.optimizer = get_optimizer(self.model.parameters(), lr)
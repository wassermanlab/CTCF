import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import shutil
from sklearn.metrics import (
    average_precision_score, matthews_corrcoef, precision_recall_curve,
    roc_curve, roc_auc_score
)
from time import time
import torch

from utils.io import write
from utils.pytorchtools import EarlyStopping

class Trainer(object):
    """
    Adapted from:
    https://selene.flatironinstitute.org/selene.html#trainmodel

    This class ties together the various objects and methods needed to train
    and validate a model.

    {Trainer} saves a checkpoint model after every epoch as well as a
    best-performing model to `output_dir`.

    {Trainer} also outputs the training and validation losses to monitor if the
    model is still improving of if there are signs of overfitting, etc.

    Parameters
    ----------
    architecture : str
        The model architecture to train.
    feature : str
        The name of the feature to train.
    generators : dict
        A dictionary that maps the `train`, `validation` and `test` steps
        to their corresponding `torch.utils.data.DataLoader` instances.
    lr: float, optional
        Default is 0.003. Sets the learning rate.
    max_epochs : int, optional
        Default is 100. The maximum number of epochs to iterate over.
    output_dir : str, optional
        Default is current working directory. The output directory to save
        model checkpoints and logs in.
    verbose: bool, optional
        Default is `False`.

    Attributes
    ----------
    architecture : str
        The name of the model architecture.
    criterion : `torch.nn._Loss`
        The loss function to optimize.
    feature : str
        The name of the feature.
    generators : dict
        The `torch.utils.data.DataLoader` for the `train`, `validation` and
        `test` sets.
    lr : float
        The learning rate.
    max_epochs : int
        The maximum number of epochs to iterate over.
    model : `torch.nn.Module`
        The model to train.
    optimizer : `torch.optim.Optimizer`
        The optimizer to minimize loss with.
    output_dir : str
        The directory to save the model and files.
    state_dict : str
        Path to the model's `state_dict`.
    """

    def __init__(
        self, architecture, feature, generators, lr=0.003, max_epochs=100,
        output_dir="./", verbose=False
    ):
        """
        Constructs a new `Trainer` object.
        """

        self.architecture = architecture
        self.feature = feature
        self.generators = generators
        self.lr = lr
        self.max_epochs = max_epochs
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._verbose = verbose

        # CUDA
        self._use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if self._use_cuda else "cpu")

        # Losses
        self.train_losses = []
        self.validation_losses = []

        # Predictions
        self.predictions = None
        self.labels = None

        # Metrics
        self.metrics = {}

        # Initialize model
        self.__initialize_model(architecture, lr)

        if self._verbose:
            write(
                None,
                "Training parameters: batch size {0}, "
                "maximum number of epochs: {1}, "
                "use cuda: {2}".format(
                    generators["train"].__dict__["batch_size"],
                    self.max_epochs,
                    self._use_cuda
                )
            )

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
            from models.danq import (
                DanQ as model_class, get_criterion, get_optimizer
            )

        data, labels = next(iter(self.generators["train"]))
        
        self.model = model_class(data.shape[-1], 1).to(self._device)
        self.criterion = get_criterion()
        self.optimizer = get_optimizer(self.model.parameters(), lr)
        self.state_dict = os.path.join(self.output_dir, "model.pth.tar")

    def train_and_validate(self):
        """Trains the model and measures validation performance."""

        early_stopping = EarlyStopping(20, True, path=self.state_dict)
        epoch_len = len(str(self.max_epochs))

        for epoch in range(1, self.max_epochs + 1):

            # Train
            t_time = time()
            t_losses = self.train()
            t_loss = np.average(t_losses)
            t_time = time() - t_time
            self.train_losses.append(t_losses)

            # Validate
            v_time = time()
            v_losses = self.validate()
            v_loss = np.average(v_losses)
            v_time = time() - v_time
            self.validation_losses.append(v_losses)

            if self._verbose:
                write(
                    None,
                    (f'[{epoch:>{epoch_len}}/{self.max_epochs:>{epoch_len}}] '
                    +f'train_loss: {t_loss:.5f} ({t_time:.3f} sec) '
                    +f'valid_loss: {v_loss:.5f} ({v_time:.3f} sec)')
                )

            # EarlyStopping needs to check if the validation loss has decresed, 
            # and if it has, it will save the current model.
            early_stopping(v_loss, self.model)
            if early_stopping.early_stop:
                if self._verbose:
                    write(None, "Stop!!!")
                break

    def train(self):
        """
        Returns
        -------
        list
            All losses.
        """

        self.model.train() # set the model in train mode
        losses = []

        for inputs, targets in self.generators["train"]:

            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            # Zero existing gradients so they don't add up
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Back-propagate and optimize
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return(losses)

    def validate(self):
        """
        Returns
        -------
        list
            All losses.
        """

        self.model.eval() # set the model in evaluation mode
        losses = []

        for inputs, targets in self.generators["validation"]:

            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                losses.append(loss.item())

        return(losses)

    def visualize_loss(self):

        # Losses to DataFrame
        data = []
        for i in range(len(self.train_losses)):
            for j in range(len(self.train_losses[i])):
                data.append(["train", i+1, j+1, self.train_losses[i][j]])
        for i in range(len(self.validation_losses)):
            for j in range(len(self.validation_losses[i])):
                data.append(
                    ["validation", i+1, j+1, self.validation_losses[i][j]]
                )
        df = pd.DataFrame(data, columns=["Mode", "Epoch", "Batch", "Loss"])

        # Seaborn aesthetics
        sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
        sns.set_palette(sns.color_palette(["#1965B0", "#DC050C"]))

        # Plot losses
        g = sns.lineplot(x="Epoch", y="Loss", hue="Mode", data=df)

        # Plot best epoch (i.e. lowest validation loss)
        best_epoch = df[(df.Mode == "validation")][["Epoch", "Loss"]]\
            .groupby("Epoch").mean().idxmin()
        g.axvline(
            int(best_epoch), linestyle=":", color="dimgray", label="best epoch"
        )

        # Plot legend
        g.legend_.remove()
        handles, labels = g.axes.get_legend_handles_labels()
        plt.legend(handles, labels, frameon=False)

        # Fine-tune plot
        g.set(xlim=(0, int(df["Epoch"].max()) + 1))
        g.set(ylim=(0, 0.5))

        # Remove spines
        sns.despine()

        # Save & close
        fig = g.get_figure()
        fig.savefig(os.path.join(self.output_dir, "loss.png"))
        plt.close(fig)

    def test(self):

        # Load the best model
        self.model.load_state_dict(torch.load(self.state_dict))
        self.model.eval() # set the model in evaluation mode

        for inputs, targets in self.generators["test"]:

            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():

                # Forward pass
                outputs = torch.sigmoid(self.model(inputs))

                if self.predictions is None and self.labels is None:
                    self.predictions = outputs.data.cpu().numpy()
                    self.labels = targets.data.cpu().numpy()
                else:
                    self.predictions = np.append(
                        self.predictions, outputs.data.cpu().numpy(), axis=0
                    )
                    self.labels = np.append(
                        self.labels, targets.data.cpu().numpy(), axis=0
                    )

    def compute_performance_metrics(self):

        # Metrics
        metrics = ["AUCPR", "AUCROC", "MCC"]

        # Flatten predictions/labels
        predictions = self.predictions.flatten()
        labels = self.labels.flatten()

        # Losses to DataFrame
        for metric in metrics:
            if metric == "AUCPR":
                score = average_precision_score(labels, predictions)
                self.metrics.setdefault(metric, score)
                prec, recall, _ = precision_recall_curve(labels, predictions)
                # i.e. precision = 0, recall = 1
                prec = np.insert(prec, 0, 0., axis=0)
                recall = np.insert(recall, 0, 1., axis=0)
                data = list(zip(recall, prec))
                self.__visualize_metric(data, ["Recall", "Precision"], metric)
            elif metric == "AUCROC":
                score = roc_auc_score(labels, predictions)
                self.metrics.setdefault(metric, score)
                fpr, tpr, _ = roc_curve(labels, predictions)
                data = list(zip(fpr, tpr))
                self.__visualize_metric(data, ["Fpr", "Tpr"], metric)
            elif metric == "MCC":
                score = matthews_corrcoef(labels, np.rint(predictions))
                self.metrics.setdefault(metric, score)
    
        if self._verbose:
            write(
                None,
                (f'Final performance metrics: '
                +f'AUCROC: {self.metrics["AUCROC"]:.5f}, '
                +f'AUCPR: {self.metrics["AUCPR"]:.5f}, '
                +f'MCC: {self.metrics["MCC"]:.5f}')
            )

    def __visualize_metric(self, data, labels, metric):

        # Metric to DataFrame
        df = pd.DataFrame(data, columns=labels)

        # Seaborn aesthetics
        sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
        sns.set_palette(sns.color_palette(["#1965B0"]))

        # Plot metric
        kwargs = dict(estimator=None, ci=None)
        g = sns.lineplot(x=labels[0], y=labels[1], data=df, **kwargs)

        # Add metric score
        kwargs = dict(horizontalalignment="center", verticalalignment="center")
        plt.text(.5, 0, "%s = %.5f" % (metric, self.metrics[metric]), **kwargs)

        # Remove spines
        sns.despine()

        # Save & close
        fig = g.get_figure()
        fig.savefig(os.path.join(self.output_dir, "%s.png" % metric))
        plt.close(fig)

    def save(self):

        # Remove non-serializable keys
        trainer_dict = {}
        for k, v in self.__dict__.items():
            try:
                json.dumps(v)
                is_JSON_serializable = True
            except:
                is_JSON_serializable = False
            if is_JSON_serializable:
                trainer_dict.setdefault(k, v)
            else:
                trainer_dict.setdefault(k, None)

        # Write JSON
        json_file = os.path.join(self.output_dir, "model.json")
        fh = open(json_file, "w")
        json.dump(
            trainer_dict, fh, sort_keys=True, indent=4, separators=(",", ": ")
        )
        fh.close()
        write(None, "`Trainer` object saved!\n%s\n" % json_file)


import numpy as np
import os
import shutil
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from time import strftime
from time import time
import torch
# from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.io import write
from .utils import PerformanceMetrics

# torch.backends.cudnn.benchmark = True

# Default metrics
default_metrics = dict(
    roc_auc=roc_auc_score,
    average_precision=average_precision_score,
    m_corr_coef=matthews_corrcoef
)

class Trainer(object):
    """
    Adapted from:
    https://selene.flatironinstitute.org/selene.html#trainmodel

    This class ties together the various objects and methods needed to train
    and validate a model.

    {TrainModel} saves a checkpoint model (overwriting it after
    `save_checkpoint_every_n_steps`) as well as a best-performing model
    (overwriting it after `report_stats_every_n_steps` if the latest validation
    performance is better than the previous best-performing model) to
    `output_dir`.

    {TrainModel} also outputs 2 files that can be used to monitor training
    as Selene runs: `selene_sdk.train_model.train.txt` (training loss) and
    `selene_sdk.train_model.validation.txt` (validation loss & average
    ROC AUC). The columns in these files can be used to quickly visualize
    training history (e.g. you can use `matplotlib`, `plt.plot(auc_list)`)
    and see, for example, whether the model is still improving, if there
    are signs of overfitting, etc.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    criterion : torch.nn._Loss
        The loss function to optimize.
    optimizer : torch.optim.Optimizer
        The optimizer to minimize loss with.
    feature_index : dict
        A dictionary that maps feature indices (`int`) to names (`int`).
    generators : dict
        A dictionary that maps the `train`, `validation` and `test` steps
        to `torch.utils.data.DataLoader` instances.
    cpu_n_threads : int, optional
        Default is 1. Sets the number of OpenMP threads used for CPU
        operations.
    max_steps : int, optional
        Default is 10000. The maximum number of mini-batches to iterate
        over.
    output_dir : str, optional
        Default is current working directory. The output directory to save
        model checkpoints and logs in.
    report_stats_every_n_steps : int
        The frequency with which to report summary statistics. You can set
        this value to be equivalent to a training epoch
        (`n_steps * batch_size`) being the total number of samples seen by
        the model so far. Selene evaluates the model on the validation
        dataset every `report_stats_every_n_steps` and, if the model obtains
        the best performance so far (based on the user-specified loss function),
        Selene saves the model state to a file called `best_model.pth.tar` in
        `output_dir`.
    save_checkpoint_every_n_steps : int or None, optional
        Default is 1000. If None, set to the same value as
        `report_stats_every_n_steps`
    verbose: bool, optional
        Default is `False`.

    Attributes
    ----------
    feature_index : dict
        The names of each feature.
    generator : torch.utils.data.DataLoader
        The generator for the `train`, `validation` and `test` sets.
    loss_criterion : torch.nn._Loss
        The loss function to optimize.
    model : torch.nn.Module
        The model to train.
    optimizer_class : torch.optim.Optimizer
        The optimizer to minimize loss with.
    max_steps : int
        The maximum number of mini-batches to iterate over.
    metrics : dict
        A dictionary that maps metric names (`str`) to metric functions. By
        default, this contains `"roc_auc"`, which maps to
        `sklearn.metrics.roc_auc_score`, `"average_precision"`, which maps to
        `sklearn.metrics.average_precision_score`, and `"m_corr_coef"`, which
        maps to `sklearn.metrics.matthews_corrcoef`.
    output_dir : str
        The directory to save model checkpoints and logs.
    nth_step_report_stats : int
        The frequency with which to report summary statistics.
    nth_step_save_checkpoint : int
        The frequency with which to save a model checkpoint.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        feature_index,
        generators,
        cpu_n_threads=1,
        max_steps=10000,
        metrics=default_metrics,
        output_dir="./",
        report_stats_every_n_steps=1000,
        save_checkpoint_every_n_steps=1000,
        verbose=False
    ):
        """
        Constructs a new `TrainModel` object.
        """
        torch.set_num_threads(cpu_n_threads)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.generators = generators

        # Optional
        self.max_steps = max_steps
        self.nth_step_report_stats = report_stats_every_n_steps
        self.nth_step_save_checkpoint = None
        if not save_checkpoint_every_n_steps:
            self.nth_step_save_checkpoint = report_stats_every_n_steps
        else:
            self.nth_step_save_checkpoint = save_checkpoint_every_n_steps
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.verbose = verbose

        # CUDA
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = torch.device("cuda:0")
            self.model.cuda()
            self.criterion.cuda()
        else:
            self.device = torch.device("cpu")

        # Metrics
        self._validation_metrics = PerformanceMetrics(
            feature_index, 1, metrics
        )
        self._test_metrics = PerformanceMetrics(
            feature_index, 1, metrics
        )

        # Extra
        self._min_loss = float("inf")
        self._targets = dict(validation=[], test=[])
        for key in self._targets:
            for inputs, targets in self.generators[key]:
                self._targets[key] += targets
            self._targets[key] = np.expand_dims(
                np.array(self._targets[key]), axis=1
            )

        if self.verbose:
            write(
                None,
                "Training parameters set: batch size {0}, "
                "number of steps per epoch: {1}, "
                "maximum number of steps: {2}, "
                "use cuda: {3}".format(
                    generators["train"].__dict__["batch_size"],
                    self.nth_step_report_stats,
                    self.max_steps,
                    self.use_cuda
                )
            )


#     def _create_validation_set(self, n_samples=None):
#         """
#         Generates the set of validation examples.
#         Parameters
#         ----------
#         n_samples : int or None, optional
#             Default is `None`. The size of the validation set. If `None`,
#             will use all validation examples in the sampler.
#         """
#         logger.info("Creating validation dataset.")
#         t_i = time()
#         # self._validation_data, self._all_validation_targets = \
#         #     self.sampler.get_validation_set(
#         #         self.batch_size, n_samples=n_samples)
#         t_f = time()
#         logger.info(("{0} s to load {1} validation examples ({2} validation "
#                      "batches) to evaluate after each training step.").format(
#                       t_f - t_i,
#                       len(self._validation_data) * self.batch_size,
#                       len(self._validation_data)))

#     def create_test_set(self):
#         """
#         Loads the set of test samples.
#         We do not create the test set in the `TrainModel` object until
#         this method is called, so that we avoid having to load it into
#         memory until the model has been trained and is ready to be
#         evaluated.
#         """
#         logger.info("Creating test dataset.")
#         t_i = time()
#         # self._test_data, self._all_test_targets = \
#         #     self.sampler.get_test_set(
#         #         self.batch_size, n_samples=self._n_test_samples)
#         t_f = time()
#         logger.info(("{0} s to load {1} test examples ({2} test batches) "
#                      "to evaluate after all training steps.").format(
#                       t_f - t_i,
#                       len(self._test_data) * self.batch_size,
#                       len(self._test_data)))
#         np.savez_compressed(
#             os.path.join(self.output_dir, "test_targets.npz"),
#             data=self._all_test_targets)

    def train_and_validate(self):
        """
        Trains the model and measures validation performance.
        """
        training_times = []
        min_loss = self._min_loss
        scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=16, verbose=True, factor=0.8
        )

        for epoch in range(10):

            # Train
            t = time()
            train_loss = self.train()
            training_times.append(time() - t)

            # Checkpoint
            checkpoint_dict = {
                "epoch": epoch,
                "arch": self.model.__class__.__name__,
                "state_dict": self.model.state_dict(),
                "min_loss": min_loss,
                "optimizer": self.optimizer.state_dict()
            }
            write(
                None,
                "Epoch {0}: train loss is `{1}`.".format(epoch, train_loss)
            )
            checkpoint_filename = "checkpoint-{0}".format(
                strftime("%m%d%H%M%S")
            )
            self.__save_checkpoint(
                checkpoint_dict, False, prefix=checkpoint_filename
            )
            write(
                None,
                "Checkpoint `{0}.pth.tar` saved.".format(checkpoint_filename)
            )

            # # Validate
            # valid_scores = self.validate()
            # validation_loss = valid_scores["loss"]
            # exit(0)
            # self._train_logger.info(train_loss)
            # to_log = [str(validation_loss)]
            # for k in sorted(self._validation_metrics.metrics.keys()):
            #     if k in valid_scores and valid_scores[k]:
            #         to_log.append(str(valid_scores[k]))
            #     else:
            #         to_log.append("NA")
            # self._validation_logger.info("\t".join(to_log))
            # scheduler.step(math.ceil(validation_loss * 1000.0) / 1000.0)

            # if validation_loss < min_loss:
            #     min_loss = validation_loss
            #     self.__save_checkpoint({
            #         "step": step,
            #         "arch": self.model.__class__.__name__,
            #         "state_dict": self.model.state_dict(),
            #         "min_loss": min_loss,
            #         "optimizer": self.optimizer.state_dict()}, True)
            #     logger.debug("Updating `best_model.pth.tar`")
            # logger.info("training loss: {0}".format(train_loss))
            # logger.info("validation loss: {0}".format(validation_loss))

                # Logging training and validation on same line requires 2 parsers or more complex parser.
                # Separate logging of train/validate is just a grep for validation/train and then same parser.
        # self.sampler.save_dataset_to_file("train", close_filehandle=True)

    def train(self):
        """
        Returns
        -------
        float
            The average loss.
        """
        # self.model.train()
        # acc_loss = []

        # for inputs, targets in self.generators["train"]:
        #     inputs = inputs.to(device=self.device, dtype=torch.float)
        #     targets = targets.to(device=self.device, dtype=torch.float)
        #     self.optimizer.zero_grad()
        #     predictions = self.model(inputs.transpose(1, 2))
        #     loss = self.criterion(predictions, targets)
        #     loss.backward()
        #     self.optimizer.step()
        #     batch_losses.append()

        # print(batch_losses)
        # exit(0)

        # return(sum(batch_losses) / len(self.generators["train"]))

        self.model.train() #tell model explicitly that we train
        running_loss = 0.0
        for seqs, labels in self.generators["train"]:
            x = seqs.to(self.device, dtype=torch.float) #the input here is (batch_size, 4, 200)
            labels = labels.to(self.device, dtype=torch.float)
            #zero the existing gradients so they don't add up
            self.optimizer.zero_grad()
            # Forward pass
            outputs = self.model(x.transpose(1, 2))
            loss = self.criterion(outputs, labels) 
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        #save training loss 
        return(running_loss / len(self.generators["train"]))



    def __evaluate_generator(self, generator):
        """
        Makes predictions for some labeled input data.

        Parameters
        ----------
        generator : `torch.utils.data.DataLoader`
            A `torch.utils.data.DataLoader` instance.

        Returns
        -------
        tuple(float, list(numpy.ndarray))
            Returns the average loss, and the list of all predictions.
        """
        self.model.eval()
        batch_losses = []
        batch_predictions = []

        for inputs, targets in generator:
            inputs = inputs.to(device=self.device, dtype=torch.float)
            targets = targets.to(device=self.device, dtype=torch.float)
            with torch.no_grad():
                predictions = self.model(inputs.transpose(1, 2))
                loss = self.criterion(predictions, targets)
                batch_predictions.append(predictions.data.cpu().numpy())
                batch_losses.append(loss.item())

        return(np.average(batch_losses), np.vstack(batch_predictions))

    def validate(self):
        """
        Measures model validation performance.

        Returns
        -------
        dict
            A dictionary, where keys are the names of the loss metrics,
            and the values are the average value for that metric over
            the validation set.
        """
        validation_loss, batch_predictions = self.__evaluate_generator(
            self.generators["validation"]
        )
        average_scores = self._validation_metrics.update(
            batch_predictions, np.array(self._targets["validation"])
        )
        for name, score in average_scores.items():
            write(None, "validation {0}: {1}".format(name, score))
        average_scores["loss"] = validation_loss

        return(average_scores)

#     def evaluate(self):
#         """
#         Measures the model test performance.
#         Returns
#         -------
#         dict
#             A dictionary, where keys are the names of the loss metrics,
#             and the values are the average value for that metric over
#             the test set.
#         """
#         if self._test_data is None:
#             self.create_test_set()
#         average_loss, all_predictions = self._evaluate_on_data(
#             self._test_data)

#         average_scores = self._test_metrics.update(all_predictions,
#                                                    self._all_test_targets)
#         np.savez_compressed(
#             os.path.join(self.output_dir, "test_predictions.npz"),
#             data=all_predictions)

#         for name, score in average_scores.items():
#             logger.info("test {0}: {1}".format(name, score))

#         test_performance = os.path.join(
#             self.output_dir, "test_performance.txt")
#         feature_scores_dict = self._test_metrics.write_feature_scores_to_file(
#             test_performance)

#         average_scores["loss"] = average_loss

#         self._test_metrics.visualize(
#             all_predictions, self._all_test_targets, self.output_dir)

#         return (average_scores, feature_scores_dict)

    def __save_checkpoint(self, state, is_best, prefix="checkpoint"):
        """
        Saves snapshot of the model state to file. Will save a checkpoint with
        name `<prefix>.pth.tar` and, if this is the model's best performance
        so far, will save the state to a `best_model.pth.tar` file as well.

        Models are saved in the state dictionary format. This is a more stable
        format compared to saving the whole model (which is another option sup-
        ported by PyTorch). Note that we do save a number of additional parame-
        ters in the dictionary and that the actual `model.state_dict()` is
        stored in the `state_dict` key of the dictionary loaded by `torch.load`.

        See: https://pytorch.org/docs/stable/notes/serialization.html for more
        information about how models are saved in PyTorch.

        Parameters
        ----------
        state : dict
            Information about the state of the model. Note that this is
            not `model.state_dict()`, but rather, a dictionary containing
            keys that can be used for continued training in Selene
            _in addition_ to a key `state_dict` that contains
            `model.state_dict()`.
        is_best : bool
            Is this the model's best performance so far?
        prefix : str, optional
            Default is "checkpoint". Specify the checkpoint prefix. Will append
            a file extension to the end (e.g. `checkpoint.pth.tar`).

        Returns
        -------
        None
        """
        file_name = os.path.join(self.output_dir, "%s.pth.tar" % prefix)
        torch.save(state, file_name)
        if is_best:
            best_file = os.path.join(self.output_dir, "best_model.pth.tar")
            shutil.copyfile(file_name, best_file)

def initialize_model(architecture, sequence_length, lr=0.001):
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

    if architecture == "danq":
        from .models.danq import (
            DanQ as model_class, get_criterion, get_optimizer
        )
    # if architecture == "deeperdeepsea":
    #     from .models.deeperdeepsea import (
    #         DeeperDeepSEA as model_class, get_loss_criterion, get_optimizer
    #     )
    # if architecture == "deepsea":
    #     from .models.deepsea import (
    #         DeepSEA as model_class, get_loss_criterion, get_optimizer
    #     )
    # if architecture == "heartenn":
    #     from .models.heartenn import (
    #         HeartENN as model_class, get_loss_criterion, get_optimizer
    #     )

    model = model_class(sequence_length, 1)
    # __is_lua_trained_model(model)
    criterion = get_criterion()
    optimizer = get_optimizer(model.parameters(), lr)

    return(model, criterion, optimizer)

# def __is_lua_trained_model(model):

#     if hasattr(model, "from_lua"):
#         return(model.from_lua)

#     from .utils.multi_model_wrapper import MultiModelWrapper

#     check_model = model
#     if hasattr(model, "model"):
#         check_model = model.model
#     elif type(model) == MultiModelWrapper and hasattr(model, "sub_models"):
#         check_model = model.sub_models[0]
#     setattr(model, "from_lua", False)
#     setattr(check_model, "from_lua", False)
#     for m in check_model.modules():
#         if "Conv2d" in m.__class__.__name__:
#             setattr(model, "from_lua", True)
#             setattr(check_model, "from_lua", True)

#     return(model.from_lua)
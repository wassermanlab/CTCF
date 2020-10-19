"""
DanQ architecture (Quang & Xie, 2016).
"""

import math
import numpy as np
import torch
import torch.nn as nn

class DanQ(nn.Module):
    """
    Parameters
    ----------
    sequence_length : int
        The length of the sequences on which the model trains and and makes
        predictions.
    n_targets : int
        The number of targets (classes) to predict.

    Attributes
    ----------
    nnet : torch.nn.Sequential
        Some description.
    bdlstm : torch.nn.Sequential
        Some description.
    classifier : torch.nn.Sequential
        The linear classifier and sigmoid transformation components of the
        model.
    """

    def __init__(self, sequence_length, n_targets):
        super(DanQ, self).__init__()

        self.nnet = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=26),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=13, stride=13),
            nn.Dropout(0.2)
        )

        self.bdlstm = nn.Sequential(
            nn.LSTM(
                320, 320, num_layers=1, batch_first=True, bidirectional=True
            )
        )

        self._n_channels = math.floor((sequence_length - 25) / 13)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._n_channels * 640, 925),
            nn.ReLU(inplace=True),
            nn.Linear(925, n_targets),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward propagation of a batch."""
        out = self.nnet(x)
        reshape_out = out.transpose(0, 1).transpose(0, 2)
        out, _ = self.bdlstm(reshape_out)
        out = out.transpose(0, 1)
        reshape_out = out.contiguous().view(
            out.size(0), 640 * self._n_channels)

        return(self.classifier(reshape_out))

def get_criterion():
    """
    Specify the appropriate loss function (criterion) for this model.

    Returns
    -------
    torch.nn._Loss
    """
    # return(nn.BCELoss())
    return(nn.BCEWithLogitsLoss())

# def get_optimizer(lr=0.001):
def get_optimizer(params, lr=0.001):
    # return(torch.optim.RMSprop, {"lr": lr})
    return(torch.optim.Adam(params, lr=lr))

"""
DanQ architecture (Quang & Xie, 2016).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DanQ(nn.Module):
    """
    Parameters
    ----------
    sequence_length : int
        The length of the sequences on which the model trains and and
        makes predictions.
    n_targets : int
        The number of targets (classes) to predict.

    Attributes
    ----------
    nnet : torch.nn.Sequential
        Some description.
    bilstm : torch.nn.Sequential
        Some description.
    classifier : torch.nn.Sequential
        The linear classifier component of the model.
    """

    def __init__(self, sequence_length, n_targets):
        super(DanQ, self).__init__()

        self._n_channels = math.floor((sequence_length - 25) / 13)

        self.nnet = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=26),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=13, stride=13),
            nn.Dropout(0.2)
        )

        self.bilstm = nn.Sequential(
            nn.LSTM(
                320, 320, num_layers=2, batch_first=True, bidirectional=True,
                dropout=0.5
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(self._n_channels * 640, 925),
            nn.ReLU(inplace=True),
            nn.Linear(925, n_targets),
        )

    def forward(self, x):
        out = self.nnet(x)
        reshape_out = torch.transpose(out, 1, 2)
        out, _ = self.bilstm(reshape_out)
        reshape_out = out.contiguous().view(-1, self._n_channels * 640)

        return(self.classifier(reshape_out))
    
def get_criterion():
    """
    Specify the appropriate loss function (criterion) for this model.

    Returns
    -------
    torch.nn._Loss
    """
    return(nn.BCEWithLogitsLoss())

def get_optimizer(params, lr=0.003):
    return(torch.optim.Adam(params, lr=lr))
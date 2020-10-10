"""
DanQ architecture (Quang & Xie, 2016).
"""

import math
import numpy as np
import torch
import torch.nn as nn

class DanQ(nn.Module):
    def __init__(self, sequence_length, num_classes):
        super(DanQ, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26)
        self.Maxpool = nn.MaxPool1d(kernel_size=13, stride=13)
        self.Drop1 = nn.Dropout(p=0.2)
        self.BiLSTM = nn.LSTM(input_size=320, hidden_size=320, num_layers=2,
                                 batch_first=True,
                                 dropout=0.5,
                                 bidirectional=True)
        self.Linear1 = nn.Linear(13*640, 925)
        self.Linear2 = nn.Linear(925, num_classes)

    def forward(self, x):
        x = self.Conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x_x = torch.transpose(x, 1, 2)
        x, (h_n,h_c) = self.BiLSTM(x_x)
        x = x.contiguous().view(-1, 13*640)
        x = self.Linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.Linear2(x)

        return(x)

def get_criterion():
    """
    Specify the appropriate loss function (criterion) for this model.

    Returns
    -------
    torch.nn._Loss
    """
    return(nn.BCEWithLogitsLoss())

def get_optimizer(params, lr=0.01):
    return(torch.optim.Adam(params, lr=lr))

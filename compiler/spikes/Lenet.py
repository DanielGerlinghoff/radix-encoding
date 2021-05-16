import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys

from spikes import Quantize as Q


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layer_list = nn.ModuleList()

        self.layer_list.append(Q.Conv2dPrune(1, 6, 5, bias=False))
        self.layer_list.append(nn.ReLU())
        self.layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer_list.append(Q.Conv2dPrune(6, 16, 5, bias=False))
        self.layer_list.append(nn.ReLU())
        self.layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer_list.append(Q.Conv2dPrune(16, 120, 5, bias=False))
        self.layer_list.append(nn.ReLU())

        self.layer_list.append(nn.Flatten())

        self.layer_list.append(Q.LinearPrune(120, 84, bias=False))
        self.layer_list.append(nn.ReLU())

        self.layer_list.append(Q.LinearPrune(84, 10, bias=False))
        self.layer_list.append(nn.LogSoftmax(dim=-1))

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

    conv_dim = 3
    linear_dim = 2

def LeNet():
    net = LeNet5()
    return net


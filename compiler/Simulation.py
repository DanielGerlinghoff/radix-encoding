"""
  Company:     A*STAR IHPC
  Engineer:    Gerlinghoff Daniel
  Create Date: 10/05/2021
 
  Description: Simulation of quantized neural network
 
"""

import torch
import torch.nn as nn
import copy

class Simulation(nn.Module):
    def __init__(self, layer_list):
        super().__init__()
        self.layer_list = copy.deepcopy(layer_list)

    def quantize_weights(self):
        for layer in self.layer_list:
            if hasattr(layer, "weight_qt"):
                layer.weight = nn.Parameter(layer.weight_qt.type(torch.float))
            if hasattr(layer, "bias_qt"):
                layer.bias = nn.Parameter(layer.bias_qt.type(torch.float))

    def quantize_activations(self, x, scale):
        return x.mul(2 ** scale - 1).round()

    def requantize_activations(self, x, scale_w, scale_in, scale_out):
        shift = scale_w + scale_in - scale_out
        x = x.type(torch.int)
        x = (x >> shift) + (x >> (shift - 1)).bitwise_and(torch.ones_like(x))
        return x.type(torch.float)

    def forward(self, x):
        x = self.quantize_activations(x, self.layer_list[0].hardware["act_in_scale"])
        for layer in self.layer_list:
            x = layer(x)
            if "wgt_scale" in layer.hardware.keys():
                scales = (layer.hardware["wgt_scale"], layer.hardware["act_in_scale"], layer.hardware["act_out_scale"])
                x = self.requantize_activations(x, *scales)
        return x

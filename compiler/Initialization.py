"""
  Company:     A*STAR IHPC
  Engineer:    Gerlinghoff Daniel
  Create Date: 06/05/2021

  Description: Statistically termine quantization scaling factors and store quantized
               weights and input data to memory initialization files

"""

import torch
import torch.nn as nn
import math
from spikes import Config
from spikes import Quantize as Q

class Network(nn.Module):
    def __init__(self, layer_list):
        super().__init__()
        self.layer_list  = layer_list

        self.act_res   = Config.resolution()[0]
        self.wgt_res   = Config.resolution()[1]
        self.wgt_sigma = 3

    def forward(self, x):
        for layer in self.layer_list:
            if type(layer) in [nn.Conv2d, Q.Conv2dPrune, nn.Linear, Q.LinearPrune]:
                if not hasattr(layer, "act_in_limit"): layer.act_in_limit = 0
                self.quantize_act_in(layer, x)
            x = layer(x)
            if type(layer) in [nn.Conv2d, Q.Conv2dPrune, nn.Linear, Q.LinearPrune]:
                if not hasattr(layer, "weight_qt"): self.quantize_wgt(layer)
                if not hasattr(layer, "act_out_limit"): layer.act_out_limit = 0
                self.quantize_act_out(layer, x)

    def quantize_act_in(self, layer, activation):
        layer.act_in_limit = max(layer.act_in_limit, activation.max())
        layer.act_in_scale = self.act_res - math.ceil(math.log2(layer.act_in_limit))

    def quantize_act_out(self, layer, activation):
        layer.act_out_limit = max(layer.act_out_limit, activation.max())
        layer.act_out_scale = self.act_res - math.ceil(math.log2(layer.act_out_limit))

    def quantize_wgt(self, layer):
        std, mean = torch.std_mean(layer.weight)
        limit     = abs(mean) + self.wgt_sigma * std
        scale     = self.wgt_res - math.ceil(math.log2(limit)) - 1

        weights_scaled     = layer.weight.mul(math.pow(2, scale))
        weights_quantized  = weights_scaled.round().clip(-2 ** (self.wgt_res - 1), 2 ** (self.wgt_res - 1) - 1)
        layer.weight_qt    = weights_quantized.type(torch.int)
        layer.weight_scale = scale

class Initialization:
    def __init__(self, layers, dataset, model_path):
        self.network = Network(layers)
        self.dataset = dataset

        self.network.load_state_dict(torch.load(model_path))
        self.network.eval()

    def layer_scaling_factors(self):
        with torch.no_grad():
            self.network(torch.stack([image for image, label in self.dataset]))

    def write_weight_files(self):
        # NOTE: Only if kernels fit into FPGA
        layer_conv_cnt = 0
        layer_lin_cnt  = 0
        for layer in self.network.layer_list:
            if type(layer) in [nn.Conv2d, Q.Conv2dPrune]:
                wgt_file = open(f"generated/bram_kernel_{layer_conv_cnt:02d}.mif", "w")
                layer_conv_cnt += 1
                for ch_out in range(layer.out_channels):
                    for ch_in in range(layer.in_channels):
                        kernel = layer.weight_qt[ch_out, ch_in]
                        kernel_packed = ""
                        for k in kernel.flatten():
                            value_str = "{:b}".format(int(k) & 0xffffffff)
                            if k >= 0:
                                kernel_packed += value_str.zfill(self.network.wgt_res)
                            else:
                                kernel_packed += value_str[-self.network.wgt_res:]
                        wgt_file.write(f"{kernel_packed}\n")
                wgt_file.close()

            elif type(layer) in [nn.Linear, Q.LinearPrune]:
                wgt_file = open(f"generated/bram_weight_{layer_lin_cnt:02d}.mif", "w")
                layer_lin_cnt += 1
                for ch_out in range(0, layer.out_features, layer.parallel):
                    for ch_in in range(layer.in_features):
                        weight = layer.weight_qt[ch_out:ch_out+layer.parallel, ch_in]
                        weight_packed = ""
                        for w in weight:
                            value_str = "{:b}".format(int(w) & 0xffffffff)
                            if w >= 0:
                                weight_packed = value_str.zfill(self.network.wgt_res) + weight_packed
                            else:
                                weight_packed = value_str[-self.network.wgt_res:] + weight_packed
                        wgt_file.write(f"{weight_packed}\n")
                wgt_file.close()

    def write_input_file(self, index):
        image = self.dataset[index][0]
        scale = self.network.act_res
        width = image.shape[2]

        input_scaled = image.mul(2 ** scale - 1)
        input        = input_scaled.round().clip(0, 2 ** self.network.act_res - 1).type(torch.int32)

        act_file = open("generated/bram_activation.mif", "w")
        bin_file = open("generated/bram_activation.bin", "wb")
        for bit in range(self.network.act_res - 1, -1, -1):
            for chn in range(input.shape[0]):
                for row in range(input.shape[1]):
                    data_full = input[chn, row]
                    data_bit  = data_full.bitwise_and(torch.ones_like(data_full) << bit) >> bit
                    data_str = ""
                    for val in data_bit:
                        data_str += str(int(val))
                    data_str = data_str.ljust(width, "0")
                    act_file.write(f"{data_str}\n")
                    data_bytes = bytearray()
                    for byte in range(math.ceil(len(data_str)/8)):
                        bits = data_str[byte*8:(byte+1)*8]
                        data_bytes.append(int(bits, 2))
                    bin_file.write(data_bytes)

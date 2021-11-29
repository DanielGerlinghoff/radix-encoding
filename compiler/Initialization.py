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

class Network(nn.Module):
    def __init__(self, layer_list, config):
        super().__init__()
        self.layer_list = layer_list
        self.config     = config

    def forward(self, x):
        # Fold batchnorm
        for mod in self.layer_list:
            if type(mod) in [nn.Conv2d]:
                mod_dst = mod
            elif type(mod) in [nn.BatchNorm2d]:
                mod_src = mod

                # Transfer bias and weights
                bias_new = mod_src.bias - mod_src.running_mean * mod_src.weight / torch.sqrt(mod_src.running_var)
                mod_dst.bias = nn.Parameter(bias_new)

                weight_new = mod_src.weight / torch.sqrt(mod_src.running_var)
                mod_dst.weight = nn.Parameter(mod_dst.weight.mul(weight_new.view(-1, 1, 1, 1).expand_as(mod_dst.weight)))

                # Neutralize batch normalization layer
                mod_src.weight = nn.Parameter(torch.ones_like(mod_src.weight))
                mod_src.bias = nn.Parameter(torch.zeros_like(mod_src.bias))
                mod_src.running_mean = torch.zeros_like(mod_src.running_mean)
                mod_src.running_var = torch.ones_like(mod_src.running_var)
                mod_src.eps = 0

        # Quantize weights and activations
        for layer in self.layer_list:
            if type(layer) in [nn.Conv2d, nn.Linear]:
                layer.hardware["act_in_scale"] = self.quantize_act(x)
            x = layer(x)
            if type(layer) in [nn.Conv2d, nn.Linear]:
                layer.hardware["act_out_scale"] = self.quantize_act(x)
                layer.hardware["wgt_scale"]     = self.quantize_wgt(layer)
                self.quantize_bias(layer)

    def quantize_act(self, activation):
        limit = activation.max()
        scale = self.config["res_activation"] - math.ceil(math.log2(limit))
        return scale

    def quantize_wgt(self, layer):
        std, mean = torch.std_mean(layer.weight)
        limit     = abs(mean) + self.config["sigma_weight"] * std
        scale     = self.config["res_weight"] - math.ceil(math.log2(limit)) - 1

        weights_scaled    = layer.weight.mul(math.pow(2, scale))
        weights_quantized = weights_scaled.round().clip(-2 ** (self.config["res_weight"] - 1), 2 ** (self.config["res_weight"] - 1) - 1)
        layer.weight_qt   = weights_quantized.type(torch.int)
        return scale

    def quantize_bias(self, layer):
        if type(layer) in [nn.Conv2d, nn.Linear] and layer.bias is not None:
            res_bias = self.config["res_weight"] + self.config["res_activation"] + self.config["bits_margin"]

            bias_scaled    = layer.bias.mul(math.pow(2, layer.hardware["act_in_scale"] + layer.hardware["wgt_scale"]))
            bias_quantized = bias_scaled.round().clip(-2 ** (res_bias - 1), 2 ** (res_bias - 1) - 1)
            layer.bias_qt  = bias_quantized.type(torch.int)


class Initialization:
    def __init__(self, layers, dataset, config):
        self.network = Network(layers, config)
        self.dataset = dataset
        self.config  = config

        self.proc = None
        self.mem  = None
        self.inst = None

        self.network.load_state_dict(torch.load(config["model_path"]))
        self.network.eval()

        self.dram_addr = 0

    def link(self, proc, mem, inst):
        self.proc, self.mem, self.inst = proc, mem, inst

    def layer_scaling_factors(self):
        with torch.no_grad():
            self.network(torch.stack([image for image, label in self.dataset]))

    def write_weight_files(self):
        layer_cnt = {"conv": 0, "lin": 0}
        dram_file = {"mif": open(f"generated/dram_kernel.mif", "w"),
                     "bin": open(f"generated/dram_kernel.bin", "wb")}
        def dram_file_write(data):
            dram_file["mif"].write(f"{data}\n")
            dram_file["bin"].write(self.binarize(data))
            self.dram_addr += 1
            layer.hardware["dram_length"] += 1
            return ""

        for layer in self.network.layer_list:
            layer.hardware["dram_start"]  = self.dram_addr
            layer.hardware["dram_length"] = 0

            if type(layer) in [nn.Conv2d]:
                bram_file = open(f"generated/bram_kernel_{layer_cnt['conv']:02d}.mif", "w")
                layer_cnt["conv"] += 1
                kernel_dram = ""
                kernel_dram_bits = int(math.pow(2, math.ceil(math.log2(self.config["res_weight"] * layer.kernel_size[0] * layer.kernel_size[1]))))
                for ch_out in range(layer.out_channels):
                    for ch_in in range(layer.in_channels):
                        kernel = layer.weight_qt[ch_out, ch_in]
                        kernel_bram = ""
                        for k in kernel.flatten():
                            value_str = "{:b}".format(int(k) & 0xffffffff)
                            if k >= 0:
                                kernel_bram += value_str.zfill(self.config["res_weight"])
                            else:
                                kernel_bram += value_str[-self.config["res_weight"]:]
                        bram_file.write(f"{kernel_bram}\n")
                        kernel_dram = kernel_bram.zfill(kernel_dram_bits) + kernel_dram
                        if len(kernel_dram) == self.config["dram_data_bits"]:
                            kernel_dram = dram_file_write(kernel_dram)
                if kernel_dram:
                    kernel_dram = kernel_dram.zfill(self.config["dram_data_bits"])
                    kernel_dram = dram_file_write(kernel_dram)
                bram_file.close()

            elif type(layer) in [nn.Linear]:
                bram_file = open(f"generated/bram_weight_{layer_cnt['lin']:02d}.mif", "w")
                layer_cnt['lin'] += 1
                for ch_out in range(0, layer.out_features, layer.hardware["parallel"]):
                    for ch_in in range(layer.in_features):
                        weight = layer.weight_qt[ch_out:ch_out+layer.hardware["parallel"], ch_in]
                        weight_bram = ""
                        for w in weight:
                            value_str = "{:b}".format(int(w) & 0xffffffff)
                            if w >= 0:
                                weight_bram = value_str.zfill(self.config["res_weight"]) + weight_bram
                            else:
                                weight_bram = value_str[-self.config["res_weight"]:] + weight_bram
                        weight_bram = weight_bram.zfill(self.config["dram_data_bits"])
                        bram_file.write(f"{weight_bram}\n")
                        dram_file_write(weight_bram)
                bram_file.close()

        [f.close() for f in dram_file.values()]

    def write_input_file(self, index):
        image = self.dataset[index][0]
        scale = self.config["res_activation"]
        width = image.shape[2]

        input_scaled = image.mul(2 ** scale - 1)
        input        = input_scaled.round().clip(0, 2 ** self.config["res_activation"] - 1).type(torch.int32)

        bram_file = {"mif": open("generated/bram_activation.mif", "w"), "bin": open("generated/bram_activation.bin", "wb")}
        for bit in range(self.config["res_activation"] - 1, -1, -1):
            for chn in range(input.shape[0]):
                for row in range(input.shape[1]):
                    data_full = input[chn, row]
                    data_bit  = data_full.bitwise_and(torch.ones_like(data_full) << bit) >> bit
                    data_str = ""
                    for val in data_bit:
                        data_str += str(int(val))
                    data_str = data_str.ljust(width, "0")
                    bram_file["mif"].write(f"{data_str}\n")
                    bram_file["bin"].write(self.binarize(data_str))

        [f.close() for f in bram_file.values()]

    @staticmethod
    def binarize(data_str):
        data_bytes = bytearray()
        for byte in range(math.ceil(len(data_str)/8)):
            bits = data_str[byte*8:(byte+1)*8]
            data_bytes.append(int(bits, 2))
        return data_bytes

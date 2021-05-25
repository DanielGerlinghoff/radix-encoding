"""
  Company:     A*STAR IHPC
  Engineer:    Gerlinghoff Daniel
  Create Date: 22/04/2021

  Description: Generate configuration package from PyTorch layers

"""

import torch.nn as nn
import math
from datetime import date
from collections import OrderedDict
from spikes import Config
from spikes import Quantize as Q

class BramKernel:
    def __init__(self, kernel):
        self.kernel = kernel
        self.width_rd  = 0
        self.width_wr  = 512
        self.height_rd = 0

class BramWeight:
    def __init__(self):
        self.width_rd  = 512
        self.width_wr  = 512
        self.height_rd = 0

class BramActivation:
    def __init__(self):
        self.width  = 0
        self.height = 0

class Memory:
    def __init__(self, layers, input_dim):
        self.layers          = layers._modules
        self.ker_brams_fit   = list()
        self.ker_brams_nofit = OrderedDict()
        self.wgt_brams_fit   = list()
        self.wgt_brams_nofit = BramWeight()
        self.act_brams       = [BramActivation(), BramActivation(), BramActivation(), BramActivation()]
        self.act_sizes       = [input_dim[0]]
        self.act_channels    = [input_dim[1]]

        self.memory_limit   = 0
        self.memory_usage   = 0
        self.kernel_fit     = None
        self.instr_width    = 32
        self.dram_data_bits = 512
        self.dram_addr_bits = 29
        self.bits_margin    = 4

    def generate(self):
        conv_ping_pong = 0
        lin_ping_pong = 2
        for layer in self.layers.values():
            if type(layer) in [nn.Conv2d, Q.Conv2dPrune, nn.MaxPool2d, nn.AvgPool2d]:
                self.act_brams[conv_ping_pong].width  = max(self.act_brams[conv_ping_pong].width,
                                                            self.act_sizes[-1])
                self.act_brams[conv_ping_pong].height = max(self.act_brams[conv_ping_pong].height,
                                                            self.act_sizes[-1] * self.act_channels[-1] * Config.resolution()[0])
                conv_ping_pong = 1 if conv_ping_pong == 0 else 0

                if type(layer) in [nn.Conv2d, Q.Conv2dPrune]:
                    kernel = layer.kernel_size[0]
                    if kernel not in self.ker_brams_nofit.keys():
                        self.ker_brams_nofit[kernel] = BramKernel(kernel)
                        self.ker_brams_nofit[kernel].width_rd = 2 ** math.ceil(math.log2(kernel ** 2 * Config.resolution()[1]))
                    self.ker_brams_nofit[kernel].height_rd = max(self.ker_brams_nofit[kernel].height_rd, layer.in_channels * layer.out_channels)
                    layer.ker_bram_nofit = list(self.ker_brams_nofit.keys()).index(kernel)

                    self.ker_brams_fit.append(BramKernel(kernel))
                    self.ker_brams_fit[-1].width_rd  = kernel ** 2 * Config.resolution()[1]
                    self.ker_brams_fit[-1].height_rd = layer.in_channels * layer.out_channels
                    layer.ker_bram_fit = len(self.ker_brams_fit) - 1
                    self.memory_usage += self.ker_brams_fit[-1].width_rd * self.ker_brams_fit[-1].height_rd

                    self.act_sizes.append(math.floor((self.act_sizes[-1] + 2 * layer.padding[0] - layer.kernel_size[0]) / layer.stride[0] + 1))
                    self.act_channels.append(layer.out_channels)

                elif type(layer) in (nn.MaxPool2d, nn.AvgPool2d):
                    self.act_sizes.append(math.floor((self.act_sizes[-1] + 2 * layer.padding - layer.kernel_size) / layer.stride + 1))
                    self.act_channels.append(self.act_channels[-1])

            elif type(layer) in [nn.Linear, Q.LinearPrune]:
                self.act_brams[lin_ping_pong].width  = 1
                self.act_brams[lin_ping_pong].height = max(self.act_brams[lin_ping_pong].height,
                                                           self.act_channels[-1] * Config.resolution()[0])
                lin_ping_pong = 3 if lin_ping_pong == 2 else 2

                self.wgt_brams_nofit.height_rd = \
                    max(self.wgt_brams_nofit.height_rd,
                    layer.in_features * math.ceil(layer.out_features / layer.parallel))
                layer.wgt_bram_nofit = 0

                self.wgt_brams_fit.append(BramWeight())
                self.wgt_brams_fit[-1].height_rd = layer.in_features * math.ceil(layer.out_features / layer.parallel)
                layer.wgt_bram_fit = len(self.wgt_brams_fit) - 1
                self.memory_usage += self.wgt_brams_fit[-1].width_rd * self.wgt_brams_fit[-1].height_rd

                self.act_channels.append(layer.out_features)

            elif type(layer) in [nn.Softmax, nn.LogSoftmax]:
                self.act_brams.append(BramActivation())
                self.act_brams[-1].width = sum(Config.resolution()) + self.bits_margin
                self.act_brams[-1].height = self.act_channels[-1]

        self.kernel_fit = self.memory_usage < self.memory_limit

    def write_to_file(self, instr_height, dram_height):
        pkg_file = open("generated/pkg_memory.sv", "w")
        wr       = lambda indent, line="": pkg_file.write("\t" * indent + line + "\n")
        sv_list  = lambda py_list: "'{" + ", ".join([str(v) for v in py_list]) + "}"

        wr(0, "`timescale 1ns / 1ps")
        wr(0, "//////////////////////////////////////////////////////////////////////////////////")
        wr(0, "// Company:     A*STAR IHPC")
        wr(0, "// Engineer:    Gerlinghoff Daniel")
        wr(0, "// Create Date: " + date.today().strftime("%d/%m/%Y"))
        wr(0, "//")
        wr(0, "// Description: Automatically generated package with configurations for kernel")
        wr(0, "//              and activation memories")
        wr(0, "//")
        wr(0, "//////////////////////////////////////////////////////////////////////////////////")
        wr(0)
        wr(0)
        wr(0, "package pkg_memory;")

        wr(1, "/* Kernel memory */")
        ker_num           = len(self.ker_brams_fit)
        ker_width         = [None] * ker_num
        ker_height        = [None] * ker_num
        ker_height_wr_max = 0
        for bram_i, bram in enumerate(self.ker_brams_fit):
            ker_width[bram_i]  = bram.width_rd
            ker_height[bram_i] = bram.height_rd
        wgt_num           = len(self.wgt_brams_fit)
        wgt_width         = [None] * wgt_num
        wgt_height        = [None] * wgt_num
        for bram_i, bram in enumerate(self.wgt_brams_fit):
            wgt_width[bram_i]  = self.dram_data_bits
            wgt_height[bram_i] = bram.height_rd
        if not self.kernel_fit:
            ker_num    = len(self.ker_brams_nofit)
            ker_width  = [None] * ker_num
            ker_height = [None] * ker_num
            ker_init   = ["\"\""] * ker_num
            for bram_i, bram in enumerate(self.ker_brams_nofit.values()):
                ker_width[bram_i]  = bram.width_rd
                ker_height[bram_i] = bram.height_rd
                ker_height_wr_max  = max(ker_height_wr_max, bram.height_rd * bram.width_rd // self.dram_data_bits)
        else:
            ker_init = [f"\"bram_kernel_{i:02d}.mif\"" for i in range(ker_num)]
        wr(1, "localparam int KER_NUM = {};".format(ker_num))
        wr(1, "localparam int KER_WIDTH [KER_NUM] = {};".format(sv_list(ker_width)))
        wr(1, "localparam int KER_WIDTH_MAX = {};".format(max(ker_width)))
        wr(1, "localparam int KER_HEIGHT [KER_NUM] = {};".format(sv_list(ker_height)))
        wr(1, "localparam int KER_HEIGHT_MAX [2] = {};".format(sv_list([max(ker_height), ker_height_wr_max])))
        wr(1, "localparam [800:1] KER_INIT [KER_NUM] = {};".format(sv_list(ker_init)))
        wr(0)

        wr(1, "/* Weight memory */")
        if not self.kernel_fit:
            wgt_num    = 1
            wgt_height = [self.wgt_brams_nofit.height_rd]
            wgt_init   = ["\"\""] * wgt_num
        else:
            wgt_init = [f"\"bram_weight_{i:02d}.mif\"" for i in range(wgt_num)]
        wr(1, "localparam int WGT_NUM = {};".format(wgt_num))
        wr(1, "localparam int WGT_HEIGHT [WGT_NUM] = {};".format(sv_list(wgt_height)))
        wr(1, "localparam int WGT_HEIGHT_MAX = {};".format(max(wgt_height)))
        wr(1, "localparam [800:1] WGT_INIT [WGT_NUM] = {};".format(sv_list(wgt_init)))
        wr(0)

        wr(1, "/* Activation memory */")
        act_num    = len(self.act_brams)
        act_width  = [None] * act_num
        act_height = [None] * act_num
        for bram_i, bram in enumerate(self.act_brams):
            act_width[bram_i]  = bram.width
            act_height[bram_i] = bram.height
        wr(1, "localparam int ACT_NUM = {};".format(act_num))
        wr(1, "localparam int ACT_WIDTH [ACT_NUM] = {};".format(sv_list(act_width)))
        wr(1, "localparam int ACT_WIDTH_MAX = {};".format(max(act_width)))
        wr(1, "localparam int ACT_HEIGHT [ACT_NUM] = {};".format(sv_list(act_height)))
        wr(1, "localparam int ACT_HEIGHT_MAX = {};".format(max(act_height)))
        wr(1, "localparam [800:1] ACT_INIT = {};".format("\"bram_activation.mif\""))
        wr(0)

        wr(1, "/* Instruction memory */")
        wr(1, "localparam int INS_WIDTH = {};".format(self.instr_width))
        wr(1, "localparam int INS_HEIGHT = {};".format(instr_height))
        wr(1, "localparam [800:1] INS_INIT = {};".format("\"bram_instruction.mif\""))
        wr(0)

        wr(1, "/* External DRAM */")
        wr(1, "localparam int DRAM_DATA_BITS = {};".format(self.dram_data_bits))
        wr(1, "localparam int DRAM_ADDR_BITS = {};".format(self.dram_addr_bits))
        wr(1, "localparam int DRAM_HEIGHT = {};".format(dram_height))
        wr(0)

        wr(0, "endpackage")
        pkg_file.close()


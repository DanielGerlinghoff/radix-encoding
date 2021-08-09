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

class Memory:
    def __init__(self, layers, config):
        self.layers         = layers._modules
        self.config         = config
        self.ker_brams_fit  = list()
        self.ker_brams_nfit = OrderedDict()
        self.wgt_brams_fit  = list()
        self.wgt_brams_nfit = {"h": 0}
        self.act_brams      = [{"w": 0, "h": 0}, {"w": 0, "h": 0}, {"w": 0, "h": 0}, {"w": 0, "h": 0}]
        self.act_sizes      = [config["input_size"]]
        self.act_channels   = [config["input_channels"]]

        self.proc = None
        self.init = None
        self.inst = None

        self.memory_usage = 0
        self.fit          = None

    def link(self, proc, init, inst):
        self.proc, self.init, self.inst = proc, init, inst

    def generate(self):
        act_2d_sel = 0
        act_1d_sel = 2
        for layer in self.layers.values():
            if type(layer) in [nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d]:
                self.act_brams[act_2d_sel]["w"] = max(self.act_brams[act_2d_sel]["w"], self.act_sizes[-1])
                self.act_brams[act_2d_sel]["h"] = max(self.act_brams[act_2d_sel]["h"], self.act_sizes[-1] * self.act_channels[-1] * self.config["res_activation"])
                act_2d_sel = 1 if act_2d_sel == 0 else 0

                if type(layer) in [nn.Conv2d]:
                    kernel = layer.kernel_size[0]
                    if kernel not in self.ker_brams_nfit.keys():
                        self.ker_brams_nfit[kernel] = {"w": 2 ** math.ceil(math.log2(kernel ** 2 * self.config["res_weight"])), "h": 0}
                    self.ker_brams_nfit[kernel]["h"] = max(self.ker_brams_nfit[kernel]["h"], layer.in_channels * layer.out_channels)
                    layer.hardware["wgt_bram_nfit"] = list(self.ker_brams_nfit.keys()).index(kernel)

                    self.ker_brams_fit.append({"w": kernel ** 2 * self.config["res_weight"],
                                               "h": layer.in_channels * layer.out_channels})
                    layer.hardware["wgt_bram_fit"] = len(self.ker_brams_fit) - 1
                    self.memory_usage += self.ker_brams_fit[-1]["w"] * self.ker_brams_fit[-1]["h"]

                    self.act_sizes.append(math.floor((self.act_sizes[-1] + 2 * layer.padding[0] - layer.kernel_size[0]) / layer.stride[0] + 1))
                    self.act_channels.append(layer.out_channels)

                elif type(layer) in (nn.MaxPool2d, nn.AvgPool2d):
                    self.act_sizes.append(math.floor((self.act_sizes[-1] + 2 * layer.padding - layer.kernel_size) / layer.stride + 1))
                    self.act_channels.append(self.act_channels[-1])

            elif type(layer) in [nn.Linear]:
                self.act_brams[act_1d_sel]["w"] = 1
                self.act_brams[act_1d_sel]["h"] = max(self.act_brams[act_1d_sel]["h"], self.act_channels[-1] * self.config["res_activation"])
                act_1d_sel = 3 if act_1d_sel == 2 else 2

                self.wgt_brams_nfit["h"] = max(self.wgt_brams_nfit["h"], layer.in_features * math.ceil(layer.out_features / layer.hardware["parallel"]))
                layer.hardware["wgt_bram_nfit"] = 0

                self.wgt_brams_fit.append({"w": self.config["dram_data_bits"],
                                           "h": layer.in_features * math.ceil(layer.out_features / layer.hardware["parallel"])})
                layer.hardware["wgt_bram_fit"] = len(self.wgt_brams_fit) - 1
                self.memory_usage += self.wgt_brams_fit[-1]["w"] * self.wgt_brams_fit[-1]["h"]

                self.act_channels.append(layer.out_features)

            elif type(layer) in [nn.Softmax, nn.LogSoftmax]:
                self.act_brams.append(dict())
                self.act_brams[-1]["w"] = self.config["res_weight"] + self.config["res_activation"] + self.config["bits_margin"]
                self.act_brams[-1]["h"] = self.act_channels[-1]

        self.fit = self.memory_usage < self.config["memory_limit"]

    def write_to_file(self):
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
        ker_num           = len(self.ker_brams_fit) if self.fit else len(self.ker_brams_nfit)
        ker_width         = [None] * ker_num
        ker_height        = [None] * ker_num
        ker_height_wr_max = 0
        if self.fit:
            ker_init = [f"\"bram_kernel_{i:02d}.mif\"" for i in range(ker_num)]
            for bram_i, bram in enumerate(self.ker_brams_fit):
                ker_width[bram_i]  = bram["w"]
                ker_height[bram_i] = bram["h"]
        else:
            ker_init   = ["\"\""] * ker_num
            for bram_i, bram in enumerate(self.ker_brams_nfit.values()):
                ker_width[bram_i]  = bram["w"]
                ker_height[bram_i] = bram["h"]
                ker_height_wr_max  = max(ker_height_wr_max, bram["h"] * bram["w"] // self.config["dram_data_bits"])
        wr(1, "localparam int KER_NUM = {};".format(ker_num))
        wr(1, "localparam int KER_WIDTH [KER_NUM] = {};".format(sv_list(ker_width)))
        wr(1, "localparam int KER_WIDTH_MAX = {};".format(max(ker_width)))
        wr(1, "localparam int KER_HEIGHT [KER_NUM] = {};".format(sv_list(ker_height)))
        wr(1, "localparam int KER_HEIGHT_MAX [2] = {};".format(sv_list([max(ker_height), ker_height_wr_max])))
        wr(1, "localparam [800:1] KER_INIT [KER_NUM] = {};".format(sv_list(ker_init)))
        wr(0)

        wr(1, "/* Weight memory */")
        wgt_num    = len(self.wgt_brams_fit) if self.fit else 1
        wgt_height = [None] * wgt_num
        if self.fit:
            wgt_init = [f"\"bram_weight_{i:02d}.mif\"" for i in range(wgt_num)]
            for bram_i, bram in enumerate(self.wgt_brams_fit):
                wgt_height[bram_i] = bram["h"]
        else:
            wgt_init   = ["\"\""] * wgt_num
            wgt_height = [self.wgt_brams_nfit["h"]]
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
            act_width[bram_i]  = bram["w"]
            act_height[bram_i] = bram["h"]
        wr(1, "localparam int ACT_NUM = {};".format(act_num))
        wr(1, "localparam int ACT_WIDTH [ACT_NUM] = {};".format(sv_list(act_width)))
        wr(1, "localparam int ACT_WIDTH_MAX = {};".format(max(act_width)))
        wr(1, "localparam int ACT_HEIGHT [ACT_NUM] = {};".format(sv_list(act_height)))
        wr(1, "localparam int ACT_HEIGHT_MAX = {};".format(max(act_height)))
        wr(1, "localparam [800:1] ACT_INIT = {};".format("\"bram_activation.mif\""))
        wr(0)

        wr(1, "/* Instruction memory */")
        wr(1, "localparam int INS_WIDTH = {};".format(self.inst.instr_width))
        wr(1, "localparam int INS_HEIGHT = {};".format(len(self.inst.instr)))
        wr(1, "localparam [800:1] INS_INIT = {};".format("\"bram_instruction.mif\""))
        wr(0)

        wr(1, "/* External DRAM */")
        wr(1, "localparam int DRAM_ENABLE = {};".format(int(not self.fit)))
        wr(1, "localparam int DRAM_DATA_BITS = {};".format(self.config["dram_data_bits"]))
        wr(1, "localparam int DRAM_ADDR_BITS = {};".format(self.config["dram_addr_bits"]))
        wr(1, "localparam int DRAM_HEIGHT = {};".format(self.init.dram_addr))
        wr(0)

        wr(0, "endpackage")
        pkg_file.close()

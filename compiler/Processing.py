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

class ConvUnit:
    def __init__(self, kernel):
        self.kernel       = kernel
        self.stride       = OrderedDict()
        self.parallel_in  = OrderedDict()
        self.parallel_out = OrderedDict()
        self.act_out_max  = None

        self.max_overhead = 0.2

    def new_layer(self, layer, act_size, first=False):
        kernel  = layer.kernel_size[0]
        padding = layer.padding[0]
        stride  = layer.stride[0]
        groups  = layer.groups
        act_in  = act_size
        act_out = math.floor((act_in + 2 * padding - kernel) / stride + 1)
        layer.in_size  = act_in
        layer.out_size = act_out

        if first: self.act_out_max = act_out
        parallel = math.floor(self.act_out_max / act_out)
        if parallel not in self.parallel_in.keys():
            self.parallel_in[parallel] = [[None, None]] * parallel
            assign_width = act_in + padding
            if assign_width % 2 == 1: assign_width = assign_width + 1
            for assign in range(parallel):
                start = padding + assign * assign_width
                end   = padding + act_in + assign * assign_width - 1
                self.parallel_in[parallel][assign] = [start, end]
        layer.parallel = parallel
        layer.compiler_parallel = list(self.parallel_in.keys()).index(parallel)
        if parallel not in self.parallel_out.keys():
            self.parallel_out[parallel] = [[None, None]] * parallel
            for assign in range(parallel):
                start = (self.parallel_in[parallel][assign][0] - padding) // stride
                end   = start + act_out - 1
                if end > self.act_out_max * (1 + self.max_overhead):
                    self.parallel_in[assign] = self.parallel_in[parallel][0:assign]
                    del self.parallel_in[parallel]
                    self.parallel_out[assign] = self.parallel_out[parallel][0:assign]
                    del self.parallel_out[parallel]
                    layer.parallel = assign
                    layer.compiler_parallel = list(self.parallel_in.keys()).index(assign)
                    break
                else:
                    self.parallel_out[parallel][assign] = [start, end]

        self.stride[stride] = True
        layer.compiler_stride = list(self.stride.keys()).index(stride)

        return act_out

class Processing:
    def __init__(self, layers, input_size):
        self.layers = layers._modules
        self.conv_units = dict()
        self.conv_units_dupl = list()
        self.act_sizes = [input_size]

        self.conv_bits_margin = 2
        self.max_duplication = 2

    def generate(self):
        layer_cnt = {"conv": 0}
        for layer in self.layers.values():
            if type(layer) is nn.Conv2d:
                layer.compiler_id = layer_cnt["conv"]
                layer_cnt["conv"] = layer_cnt["conv"] + 1
                kernel = layer.kernel_size[0]
                if kernel not in self.conv_units.keys():
                    self.conv_units[kernel] = ConvUnit(kernel)
                    act_out = self.conv_units[kernel].new_layer(layer, self.act_sizes[-1], first=True)
                else:
                    act_out = self.conv_units[kernel].new_layer(layer, self.act_sizes[-1])

                self.act_sizes.append(act_out)

            if type(layer) in (nn.MaxPool2d, nn.AvgPool2d):
                act_out = math.floor((self.act_sizes[-1] + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
                self.act_sizes.append(act_out)

    def duplicate(self):
        # TODO: Duplicate based on number of output channels
        for cu in self.conv_units.values():
            for _ in range(self.max_duplication):
                self.conv_units_dupl.append(cu)

    def write_to_file(self):
        pkg_file = open("generated/pkg_processing.sv", "w")
        wr       = lambda indent, line="": pkg_file.write("\t" * indent + line + "\n")
        sv_list  = lambda py_list: "'{" + ", ".join([str(v) for v in py_list]) + "}"

        wr(0, "`timescale 1ns / 1ps")
        wr(0, "//////////////////////////////////////////////////////////////////////////////////")
        wr(0, "// Company:     A*STAR IHPC")
        wr(0, "// Engineer:    Gerlinghoff Daniel")
        wr(0, "// Create Date: " + date.today().strftime("%d/%m/%Y"))
        wr(0, "//")
        wr(0, "// Description: Automatically generated package with config of processing units")
        wr(0, "//")
        wr(0, "//////////////////////////////////////////////////////////////////////////////////")
        wr(0)
        wr(0)
        wr(0, "package pkg_processing;")

        wr(1, "/* Convolution settings */")
        convunits        = len(self.conv_units_dupl)
        conv_size        = [0] * convunits
        ker_size         = [0] * convunits
        parallel_dim     = [None] * convunits
        parallel_dim_max = [0, 0]
        stride_dim       = [None] * convunits
        stride_dim_max   = 0
        for cu_i, cu in enumerate(self.conv_units_dupl):
            ker_size[cu_i]      = cu.kernel
            parallel_dim[cu_i]  = [len(cu.parallel_in), max(cu.parallel_in.keys())]
            parallel_dim_max[0] = max(parallel_dim_max[0], parallel_dim[cu_i][0])
            parallel_dim_max[1] = max(parallel_dim_max[1], parallel_dim[cu_i][1])
            stride_dim[cu_i]    = len(cu.stride)
            stride_dim_max      = max(stride_dim_max, stride_dim[cu_i])
            for p in cu.parallel_out.values():
                for a in p:
                    conv_size[cu_i] = max(conv_size[cu_i], a[1] + 1)
        parallel_num   = [[0] * parallel_dim_max[0]] * convunits
        parallel_width = [[0] * parallel_dim_max[0]] * convunits
        parallel_max   = [0] * convunits
        parallel_in    = [[[[0, 0] for _ in range(parallel_dim_max[1])] for _ in range(parallel_dim_max[0])] for _ in range(convunits)]
        parallel_out   = [[[[0, 0] for _ in range(parallel_dim_max[1])] for _ in range(parallel_dim_max[0])] for _ in range(convunits)]
        stride         = [[0 for _ in range(stride_dim_max)] for _ in range(convunits)]
        stride_max     = [0] * convunits
        for cu_i, cu in enumerate(self.conv_units_dupl):
            for p_i, p in enumerate(cu.parallel_in.keys()):
                parallel_num[cu_i][p_i]   = p
                parallel_width[cu_i][p_i] = cu.parallel_out[p][0][1] - cu.parallel_out[p][0][0] + 1;
                parallel_max[cu_i]        = max(parallel_max[cu_i], p)
                for a_i in range(p):
                    parallel_in[cu_i][p_i][a_i]  = cu.parallel_in[p][a_i]
                    parallel_out[cu_i][p_i][a_i] = cu.parallel_out[p][a_i]
            for s_i, s in enumerate(cu.stride):
                stride[cu_i][s_i] = s
                stride_max[cu_i]  = max(stride_max[cu_i], s)
        wr(1, "localparam int CONVUNITS = {};".format(convunits))
        wr(1, "localparam int CONV_SIZE [CONVUNITS] = {};".format(sv_list(conv_size)))
        wr(1, "localparam int CONV_SIZE_MAX = {};".format(max(conv_size)))
        wr(1, "localparam int CONV_BITS = {};".format(sum(Config.resolution()) + self.conv_bits_margin))
        wr(1, "localparam int ACT_BITS = {};".format(Config.resolution()[0]))
        wr(1, "localparam int KER_BITS = {};".format(Config.resolution()[1]))
        wr(1, "localparam int KER_SIZE [CONVUNITS] = {};".format(sv_list(ker_size)))
        wr(0)
        wr(1, "localparam int PARALLEL_DIM [CONVUNITS][2] = {};".format(sv_list([sv_list(l) for l in parallel_dim])))
        wr(1, "localparam int PARALLEL_NUM [CONVUNITS][{}] = {};".format(parallel_dim_max[0], sv_list([sv_list(l) for l in parallel_num])))
        wr(1, "localparam int PARALLEL_WIDTH [CONVUNITS][{}] = {};".format(parallel_dim_max[0], sv_list([sv_list(l) for l in parallel_width])))
        wr(1, "localparam int PARALLEL_MAX [CONVUNITS] = {};".format(sv_list(parallel_max)))
        wr(1, "localparam int PARALLEL_IN [CONVUNITS][{}][{}][2] = '{{".format(parallel_dim_max[0], parallel_dim_max[1]))
        for cu_i, cu in enumerate(parallel_in):
            wr(2, "'{")
            for p_i, p in enumerate(cu):
                wr(3, "{}{}".format(sv_list([sv_list(l) for l in p]), "," if p_i != parallel_dim_max[0] - 1 else ""))
            wr(2, "}}{}".format("," if cu_i != convunits - 1 else ""))
        wr(1, "};")
        wr(1, "localparam int PARALLEL_OUT [CONVUNITS][{}][{}][2] = '{{".format(parallel_dim_max[0], parallel_dim_max[1]))
        for cu_i, cu in enumerate(parallel_out):
            wr(2, "'{")
            for p_i, p in enumerate(cu):
                wr(3, "{}{}".format(sv_list([sv_list(l) for l in p]), "," if p_i != parallel_dim_max[0] - 1 else ""))
            wr(2, "}}{}".format("," if cu_i != convunits - 1 else ""))
        wr(1, "};")
        wr(0)
        wr(1, "localparam int STRIDE_DIM [CONVUNITS] = {};".format(sv_list(stride_dim)))
        wr(1, "localparam int STRIDE [CONVUNITS][{}] = '{{".format(stride_dim_max))
        for cu_i, cu in enumerate(stride):
            wr(2, "{}{}".format(sv_list(cu), "," if cu_i != convunits - 1 else ""))
        wr(1, "};")
        wr(1, "localparam int STRIDE_MAX [CONVUNITS] = {};".format(sv_list(stride_max)))

        wr(0)
        wr(0, "endpackage")
        pkg_file.close()

"""
  Company:     A*STAR IHPC
  Engineer:    Gerlinghoff Daniel
  Create Date: 05/05/2021

  Description: Generate instructions based on network architecture

"""

import torch.nn as nn
from spikes import Config
from spikes import Quantize as Q

class Configuration:
    def __init__(self, convunits):
        self.enable        = [0] * convunits
        self.conv_parallel = None
        self.conv_stride   = None
        self.output_mode   = None
        self.act_scale     = None

class Instructions:
    def __init__(self, layers, input_dim, processing, memory):
        self.layers     = layers._modules
        self.processing = processing
        self.memory     = memory
        self.tsteps     = Config.resolution()[0]

        self.configuration = Configuration(len(processing.conv_units_dupl))
        self.instr = list()

    def generate(self):
        ops       = {"CONF": 1, "RST": 2, "PROC": 3, "KERL": 4, "ACTL": 5, "ACTS": 6, "WAIT": 7, "END": 8}
        confs     = {"CPAR": 0, "STR": 1, "PAD": 2, "OUT": 3, "KSEL": 4, "ASEL": 5, "SCL": 6, "ASTF": 7, "ASTB": 8, "PPAR": 9}
        conds     = {"CONV": 0, "CWR": 1, "TRAN": 2}
        out_mode  = {"DEL": 3, "DIR": 2, "ADD": 1, "SFT": 0}
        ops_off   = 28
        unit_off  = 23
        confs_off = 19
        conds_off = 21
        addr_off  = 8

        instr_conf = lambda unit, conf, val: \
            self.instr.append((ops["CONF"] << ops_off) + (unit << unit_off) + (confs[conf] << confs_off) + val)
        instr_cmd = lambda op: \
            self.instr.append(ops[op] << ops_off)
        instr_en = lambda unit, en: \
            self.instr.append((ops["CONF"] << ops_off) + (unit << unit_off) + en)
        instr_mem = lambda op, unit, addr: \
            self.instr.append((ops[op] << ops_off) + (unit << unit_off) + addr)
        instr_acts = lambda unit, rd_addr, wr_addr: \
            self.instr.append((ops["ACTS"] << ops_off) + (unit << unit_off) + (wr_addr << addr_off) + rd_addr)
        instr_wait = lambda unit, cond: \
            self.instr.append((ops["WAIT"] << ops_off) + (unit << unit_off) + (conds[cond] << conds_off))

        act_ping_pong = False
        for layer in self.layers.values():
            if type(layer) in [nn.Conv2d, Q.Conv2dPrune]:
                instr_conf(31, "CPAR", layer.compiler_parallel)
                instr_conf(31, "STR", layer.compiler_stride)
                instr_conf(31, "PAD", 0)
                instr_conf(31, "KSEL", layer.compiler_id)  # NOTE: assuming kernels fit into device
                instr_conf(31, "ASEL", int(act_ping_pong))
                chn_out = 0
                while chn_out < layer.out_channels:
                    for tstep in range(self.tsteps):
                        for chn_in in range(layer.in_channels):
                            instr_cmd("RST")
                            chn_out_cu = chn_out
                            for cu_i, cu in enumerate(self.processing.conv_units_dupl):
                                if cu.kernel == layer.kernel_size[0]:
                                    instr_en(cu_i, 1)
                                    for p in range(list(cu.parallel_in.keys())[layer.compiler_parallel]):
                                        ker_addr = chn_out_cu * layer.in_channels + chn_in
                                        instr_mem("KERL", layer.compiler_id, ker_addr)
                                        if chn_out_cu < layer.out_channels - 1: chn_out_cu += 1
                                        else: break
                                    instr_en(cu_i, 0)
                                    if chn_out_cu == layer.out_channels: break
                            cu_1st = None
                            for cu_i, cu in enumerate(self.processing.conv_units_dupl):
                                if cu.kernel == layer.kernel_size[0]:
                                    if cu_1st is None: cu_1st = cu_i
                                    instr_en(cu_i, 1)
                            if not chn_in:
                                if not tstep: output_mode = out_mode["DIR"]
                                else: output_mode = out_mode["SFT"]
                            else: output_mode = out_mode["ADD"]
                            instr_conf(31, "OUT", out_mode["DEL"])
                            for pad in range(layer.padding[0]):
                                if pad == 0: instr_conf(31, "PAD", 1)
                                instr_cmd("PROC")
                                if pad == layer.kernel_size[0] - 1: instr_conf(31, "OUT", output_mode)
                                instr_wait(cu_1st, "CONV")
                            instr_conf(31, "PAD", 0)
                            instr_mem("ACTL", int(act_ping_pong), layer.in_size * (chn_in + layer.in_channels * tstep))
                            for row in range(layer.in_size):
                                instr_cmd("PROC")
                                if row == layer.kernel_size[0] - layer.padding[0] - 1: instr_conf(31, "OUT", output_mode)
                                if row < layer.in_size - 1:
                                    instr_mem("ACTL", int(act_ping_pong), row + 1 + layer.in_size * (chn_in + layer.in_channels * tstep))
                                    instr_wait(cu_1st, "CONV")
                                elif not layer.padding[0]:
                                    instr_wait(cu_1st, "CWR")
                            for pad in range(layer.padding[0]):
                                if pad == 0: instr_conf(31, "PAD", 1)
                                instr_cmd("PROC")
                                if pad < layer.padding[0] - 1: instr_wait(cu_1st, "CONV")
                                else: instr_wait(cu_1st, "CWR")
                            for cu_i, cu in enumerate(self.processing.conv_units_dupl):
                                if cu.kernel == layer.kernel_size[0]:
                                    instr_en(cu_i, 0)
                    instr_conf(31, "SCL", layer.weight_scale + layer.act_in_scale - layer.act_out_scale)
                    instr_conf(31, "ASTF", layer.out_size * layer.out_channels)
                    instr_conf(31, "ASTB", layer.out_size * (layer.out_channels * (self.tsteps - 1) - 1))
                    chn_out_cu = chn_out
                    for cu_i, cu in enumerate(self.processing.conv_units_dupl):
                        if cu.kernel == layer.kernel_size[0]:
                            for row in range(layer.out_size):
                                instr_acts(cu_i, row, row + layer.out_size * chn_out_cu)
                                instr_wait(cu_i, "TRAN")
                            chn_out_cu += layer.parallel
                    chn_out = chn_out_cu
                act_ping_pong = not act_ping_pong

            if type(layer) is nn.MaxPool2d:
                instr_conf(31, "PPAR", layer.compiler_parallel)
                instr_conf(31, "ASEL", int(act_ping_pong))
                instr_conf(31, "ASTF", layer.out_size * layer.channels)
                instr_conf(31, "ASTB", layer.out_size * (layer.channels * (self.tsteps - 1) - 1))
                for pu_i, pu in enumerate(self.processing.pool_units_dupl):
                    if pu.kernel == layer.kernel_size:
                        pu_1st = pu_i
                        break
                instr_en(pu_1st + len(self.processing.conv_units_dupl), 1)
                chn = 0
                while chn < layer.channels:
                    chn_pu = chn
                    for _ in range(layer.parallel):
                        for tstep in range(self.tsteps):
                            instr_mem("ACTL", int(act_ping_pong), layer.in_size * (chn_pu + layer.channels * tstep))
                        chn_pu += 1
                        if chn_pu == layer.channels: break
                    for row in range(layer.in_size):
                        if row % layer.kernel_size == 0:
                            instr_cmd("RST")
                            instr_acts(pu_1st + len(self.processing.conv_units_dupl), 0, row // 2 + layer.out_size * chn)
                        instr_cmd("PROC")
                        instr_conf(31, "OUT", out_mode["DIR" if (row + 1) % layer.kernel_size == 0 else "DEL"])
                        if row < layer.in_size - 1:
                            chn_pu = chn
                            for _ in range(layer.parallel):
                                for tstep in range(self.tsteps):
                                    instr_mem("ACTL", int(act_ping_pong), row + 1 + layer.in_size * (chn_pu + layer.channels * tstep))
                                chn_pu += 1
                                if chn_pu == layer.channels: break
                        else:
                            instr_wait(pu_1st, "TRAN")
                    chn = chn_pu
                instr_en(pu_1st + len(self.processing.conv_units_dupl), 0)
                act_ping_pong = not act_ping_pong

        instr_cmd("END")

    def write_to_file(self):
        data_file = open("generated/bram_instruction.mif", "w")
        for ins in self.instr:
            data_file.write(f"{ins:08x}\n")

        data_file.close()


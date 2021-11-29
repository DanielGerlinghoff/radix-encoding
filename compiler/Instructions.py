"""
  Company:     A*STAR IHPC
  Engineer:    Gerlinghoff Daniel
  Create Date: 05/05/2021

  Description: Generate instructions based on network architecture

"""

import torch.nn as nn

class Instructions:
    def __init__(self, layers, config):
        self.layers = layers._modules
        self.tsteps = config["res_activation"]

        self.proc = None
        self.init = None
        self.mem  = None

        self.instr_width = 32
        self.instr       = list()

    def link(self, proc, init, mem):
        self.proc, self.init, self.mem = proc, init, mem

    def generate(self):
        ops       = {"ENA": 0, "CONF": 1, "RST": 2, "PROC": 3, "LIN": 4, "KERD": 5, "KERL": 6, "ACTL": 7, "ACTS": 8, "WAIT": 9, "END": 10}
        confs     = {"CPAR": 0, "CSTR": 1, "CPAD": 2, "PPAR": 3, "LCHN": 4, "LRELU": 5, "OUTM": 6, "SCL": 7,
                     "KSEL": 8, "WSEL": 9, "WADR": 10, "DADR": 11, "ASELR": 12, "ASELW": 13, "ASTPF": 14, "ASTPB": 15, "ASRC": 16, "ADST": 17}
        conds     = {"CONV": 0, "CWR": 1, "TRAN": 2}
        out_mode  = {"DEL": 3, "DIR": 2, "ADD": 1, "SFT": 0}
        ops_off   = 28
        unit_off  = 23
        confs_off = 23
        conds_off = 21
        addr_off  = 8

        instr_en = lambda unit, en: \
            self.instr.append((ops["ENA"] << ops_off) + (unit << unit_off) + en)
        instr_conf = lambda conf, val: \
            self.instr.append((ops["CONF"] << ops_off) + (confs[conf] << confs_off) + val)
        instr_cmd = lambda op: \
            self.instr.append(ops[op] << ops_off)
        instr_mem = lambda op, unit, addr: \
            self.instr.append((ops[op] << ops_off) + (unit << unit_off) + addr)
        instr_acts = lambda unit, rd_addr, wr_addr: \
            self.instr.append((ops["ACTS"] << ops_off) + (unit << unit_off) + (wr_addr << addr_off) + rd_addr)
        instr_wait = lambda unit, cond: \
            self.instr.append((ops["WAIT"] << ops_off) + (unit << unit_off) + (conds[cond] << conds_off))

        conv_ping_pong = False
        lin_ping_pong = False
        for layer in self.layers.values():
            if type(layer) in [nn.Conv2d]:
                instr_conf("CPAR", layer.hardware["parallel_instr"])
                instr_conf("CSTR", layer.hardware["stride_instr"])
                instr_conf("CPAD", 0)
                if not self.mem.fit:
                    instr_conf("KSEL", layer.hardware["wgt_bram_nfit"])
                    instr_conf("DADR", layer.hardware["dram_start"])
                    instr_mem("KERD", 0, layer.hardware["dram_length"])
                else:
                    instr_conf("KSEL", layer.hardware["wgt_bram_fit"])
                instr_conf("ASELR", int(conv_ping_pong))
                instr_conf("ASELW", int(not conv_ping_pong) if layer.out_size != 1 else 2)
                chn_out = 0
                while chn_out < layer.out_channels:
                    for tstep in range(self.tsteps):
                        for chn_in in range(layer.in_channels):
                            instr_cmd("RST")
                            chn_out_cu = chn_out
                            for cu_i, cu in enumerate(self.proc.conv_units_dupl):
                                if cu.kernel == layer.kernel_size[0]:
                                    instr_en(cu_i, 1)
                                    for p in range(list(cu.parallel_in.keys())[layer.hardware["parallel_instr"]]):
                                        ker_addr = chn_out_cu * layer.in_channels + chn_in
                                        instr_mem("KERL", layer.hardware["wgt_bram_fit"] if self.mem.fit else layer.hardware["wgt_bram_nfit"], ker_addr)
                                        if chn_out_cu < layer.out_channels - 1: chn_out_cu += 1
                                        else: break
                                    instr_en(cu_i, 0)
                                    if chn_out_cu == layer.out_channels: break
                            cu_1st = None
                            for cu_i, cu in enumerate(self.proc.conv_units_dupl):
                                if cu.kernel == layer.kernel_size[0]:
                                    if cu_1st is None: cu_1st = cu_i
                                    instr_en(cu_i, 1)
                            if not chn_in:
                                if not tstep: output_mode = out_mode["DIR"]
                                else: output_mode = out_mode["SFT"]
                            else: output_mode = out_mode["ADD"]
                            instr_conf("OUTM", out_mode["DEL"])
                            for pad in range(layer.padding[0]):
                                if pad == 0: instr_conf("CPAD", 1)
                                instr_cmd("PROC")
                                if pad == layer.kernel_size[0] - 1: instr_conf("OUTM", output_mode)
                                instr_wait(cu_1st, "CONV")
                            instr_conf("CPAD", 0)
                            instr_mem("ACTL", int(conv_ping_pong), layer.in_size * (chn_in + layer.in_channels * tstep))
                            for row in range(layer.in_size):
                                instr_cmd("PROC")
                                if row == layer.kernel_size[0] - layer.padding[0] - 1: instr_conf("OUTM", output_mode)
                                if row < layer.in_size - 1:
                                    instr_mem("ACTL", int(conv_ping_pong), row + 1 + layer.in_size * (chn_in + layer.in_channels * tstep))
                                    instr_wait(cu_1st, "CONV")
                                elif not layer.padding[0]:
                                    instr_wait(cu_1st, "CWR")
                            for pad in range(layer.padding[0]):
                                if pad == 0: instr_conf("CPAD", 1)
                                instr_cmd("PROC")
                                if pad < layer.padding[0] - 1: instr_wait(cu_1st, "CONV")
                                else: instr_wait(cu_1st, "CWR")
                            for cu_i, cu in enumerate(self.proc.conv_units_dupl):
                                if cu.kernel == layer.kernel_size[0]:
                                    instr_en(cu_i, 0)
                    instr_conf("SCL", layer.hardware["wgt_scale"] + layer.hardware["act_in_scale"] - layer.hardware["act_out_scale"])
                    instr_conf("ASTPF", layer.out_size * layer.out_channels)
                    instr_conf("ASTPB", layer.out_size * (layer.out_channels * (self.tsteps - 1) - 1))
                    chn_out_cu = chn_out
                    for cu_i, cu in enumerate(self.proc.conv_units_dupl):
                        if cu.kernel == layer.kernel_size[0]:
                            for row in range(layer.out_size):
                                instr_acts(cu_i, row, row + layer.out_size * chn_out_cu)
                                instr_wait(cu_i, "TRAN")
                            chn_out_cu += layer.hardware["parallel"]
                            if chn_out_cu == layer.out_channels:
                                break
                    chn_out = chn_out_cu
                conv_ping_pong = not conv_ping_pong

            if type(layer) is nn.MaxPool2d:
                instr_conf("PPAR", layer.hardware["parallel_instr"])
                instr_conf("ASELR", int(conv_ping_pong))
                instr_conf("ASELW", int(not conv_ping_pong) if layer.out_size != 1 else 2)
                instr_conf("ASTPF", layer.out_size * layer.channels)
                instr_conf("ASTPB", layer.out_size * (layer.channels * (self.tsteps - 1) - 1))
                for pu_i, pu in enumerate(self.proc.pool_units.values()):
                    if pu.kernel == layer.kernel_size:
                        pu_1st = pu_i
                        break
                instr_en(pu_1st + len(self.proc.conv_units_dupl), 1)
                chn = 0
                while chn < layer.channels:
                    chn_pu = chn
                    for _ in range(layer.hardware["parallel"]):
                        for tstep in range(self.tsteps):
                            instr_mem("ACTL", int(conv_ping_pong), layer.in_size * (chn_pu + layer.channels * tstep))
                        chn_pu += 1
                        if chn_pu == layer.channels: break
                    for row in range(layer.in_size // layer.kernel_size * layer.kernel_size):
                        if row % layer.kernel_size == 0:
                            instr_cmd("RST")
                            instr_acts(pu_1st + len(self.proc.conv_units_dupl), 0, row // layer.kernel_size + layer.out_size * chn)
                        instr_cmd("PROC")
                        instr_conf("OUTM", out_mode["DIR" if (row + 1) % layer.kernel_size == 0 else "DEL"])
                        if row < layer.in_size - 1:
                            chn_pu = chn
                            for _ in range(layer.hardware["parallel"]):
                                for tstep in range(self.tsteps):
                                    instr_mem("ACTL", int(conv_ping_pong), row + 1 + layer.in_size * (chn_pu + layer.channels * tstep))
                                chn_pu += 1
                                if chn_pu == layer.channels: break
                        else:
                            instr_wait(pu_1st, "TRAN")
                    chn = chn_pu
                instr_en(pu_1st + len(self.proc.conv_units_dupl), 0)
                conv_ping_pong = not conv_ping_pong

            if type(layer) in [nn.Linear]:
                lu_i = len(self.proc.conv_units_dupl) + len(self.proc.pool_units)
                layer_i = list(self.layers.values()).index(layer)
                last_layer = type(self.layers[str(layer_i+1)]) in [nn.Softmax, nn.LogSoftmax]
                instr_conf("LCHN", layer.in_features)
                instr_conf("SCL", layer.hardware["wgt_scale"] + layer.hardware["act_in_scale"] - layer.hardware["act_out_scale"])
                instr_conf("LRELU", int(not last_layer))
                instr_conf("ASELR", int(lin_ping_pong) + 2)
                instr_conf("ASELW", int(not lin_ping_pong) + 2 if not last_layer else 4)
                if not self.mem.fit:
                    instr_conf("WSEL", layer.hardware["wgt_bram_nfit"])
                    instr_conf("DADR", layer.hardware["dram_start"])
                    instr_mem("KERD", 0, layer.hardware["dram_length"])
                else:
                    instr_conf("WSEL", layer.hardware["wgt_bram_fit"])
                instr_conf("ASTPF", layer.out_features if not last_layer else 1)
                instr_conf("ASTPB", (layer.out_features * (self.tsteps - 1) - 1))
                instr_en(lu_i, 1)
                chn_out = 0
                while chn_out < layer.out_features:
                    instr_cmd("RST")
                    for tstep in range(self.tsteps):
                        instr_conf("WADR", chn_out // layer.hardware["parallel"] * layer.in_features)
                        instr_conf("ASRC", tstep * layer.in_features)
                        instr_conf("ADST", chn_out)
                        instr_cmd("LIN")
                    instr_wait(lu_i, "TRAN")
                    chn_out += layer.hardware["parallel"]
                instr_en(lu_i, 0)
                lin_ping_pong = not lin_ping_pong

        instr_cmd("END")

    def write_to_file(self):
        data_file = open("generated/bram_instruction.mif", "w")
        for ins in self.instr:
            data_file.write(f"{ins:08x}\n")

        data_file.close()

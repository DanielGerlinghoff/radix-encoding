`timescale 1ns / 1ps
`default_nettype none
`include "sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 29/04/2021
//
// Description: Test of design module network
//
//////////////////////////////////////////////////////////////////////////////////


module network_tb;
    /* Clock signal */
    logic clk;
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end

    /* Module parameters */
    localparam ID_CONV = 0;
    localparam ID_KER  = 0;
    localparam ID_ACT  = 0;
    localparam KER     = pkg_processing::KER_SIZE[ID_CONV];

    /* Module input signals */
    logic dram_enable;
    logic [pkg_memory::DRAM_ADDR_BITS-1:0] dram_addr;
    logic [pkg_memory::DRAM_DATA_BITS-1:0] dram_data;

    task reset;
        #(RST_PERIOD);
        test.reset = 1;
        #(CLK_PERIOD);
        test.reset = 0;
    endtask

    task load_kernels (int addr, int cnt);
        for (int c = 0; c < cnt; c++) begin
            #(CLK_PERIOD);
            test.ker.bram_rd_en[ID_KER] = 1;
            test.ker.bram_rd_addr = addr++;
        end
        #(CLK_PERIOD);
        test.ker.bram_rd_en[ID_KER] = 0;
    endtask

    task load_activation (int addr);
        #(CLK_PERIOD);
        test.act.rd_en[ID_ACT] = 1;
        test.act.addr = addr;
        #(CLK_PERIOD);
        test.act.rd_en[ID_ACT] = 0;
    endtask

    task start_convolution;
        #(CLK_PERIOD);
        test.start = 1;
        #(CLK_PERIOD);
        test.start = 0;
    endtask

    task transfer_result (int addr_read, int addr_base);
        #(CLK_PERIOD);
        test.act.conv_rd_addr = addr_read;
        test.act.conv_rd_en[ID_CONV] = 1;
        test.act.wr_addr_base = addr_base;
        #(CLK_PERIOD);
        test.act.conv_rd_en[ID_CONV] = 0;
    endtask

    initial begin
        test.ker.bram_rd_en[0] = 0;
        test.start = 0;
        test.act.conv_rd_en[ID_CONV] = 0;

        /* Configure convolution */
        test.conf.enable[ID_CONV] = 1;
        test.conf.conv_parallel = 1;
        test.conf.conv_stride = 0;

        /* Reset convolution */
        reset();

        /* Load kernels */
        test.mem_ker.gen_bram[ID_KER].bram_i.ram[0] = {9 {8'h01}};
        test.mem_ker.gen_bram[ID_KER].bram_i.ram[1] = {9 {8'h02}};

        load_kernels(0, 2);

        /* Load activation */
        test.mem_act.gen_bram[ID_ACT].bram_i.ram[0] = {38 {3'b010}};
        test.mem_act.gen_bram[ID_ACT].bram_i.ram[1] = {38 {3'b101}};
        test.mem_act.gen_bram[ID_ACT].bram_i.ram[2] = {38 {3'b010}};
        test.mem_act.gen_bram[ID_ACT].bram_i.ram[3] = {38 {3'b101}};
        test.conf.mem_select = 0;

        load_activation(0);

        /* Start convolution */
        for (int i = 0; i < 4; i++) begin
            start_convolution();
            load_activation(i + 1);
            wait(test.finish[ID_CONV]);
            #(CLK_PERIOD/2);
            test.conf.output_mode = (i < KER - 1) ? test.conf.DEL : test.conf.DIR;
        end

        reset();
        load_activation(0);

        for (int i = 0; i < 4; i++) begin
            start_convolution();
            load_activation(i + 1);
            wait(test.finish[ID_CONV]);
            #(CLK_PERIOD/2);
            test.conf.output_mode = (i < KER - 1) ? test.conf.DEL : test.conf.ADD;
        end

        reset();
        load_activation(0);

        for (int i = 0; i < 4; i++) begin
            start_convolution();
            load_activation(i + 1);
            wait(test.finish[ID_CONV]);
            #(CLK_PERIOD/2);
            test.conf.output_mode = (i < KER - 1) ? test.conf.DEL : test.conf.SFT;
        end

        /* Move result to activation memory */
        test.conf.mem_select = 1;
        test.act.conv_scale = 4;
        test.act.conv_addr_step = '{4, 10};

        #(RST_PERIOD);
        for (int i = 0; i < 2; i++) begin
            transfer_result(i, i);
            wait(test.act.conv_transfer_finish);
            #(CLK_PERIOD/2);
        end

        #(RST_PERIOD);
        $finish();

    end

    /* Module instantiation */
    network test (
        .*
    );

endmodule


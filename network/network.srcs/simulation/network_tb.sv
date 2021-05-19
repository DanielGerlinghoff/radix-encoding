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
    localparam KER     = pkg_convolution::KER_SIZE[ID_CONV];

    /* Module input signals */
    import pkg_memory::*;

    logic                                     proc_reset, proc_start, proc_finish;
    logic                                     reset, start, finish;
    logic                                     input_en = 0;
    logic [$clog2(ACT_HEIGHT_MAX)-1:0]        input_addr;
    logic [0:ACT_WIDTH_MAX-1]                 input_data;
    logic                                     output_en = 0;
    logic [$clog2(ACT_HEIGHT[ACT_NUM-1])-1:0] output_addr;
    logic [0:ACT_WIDTH[ACT_NUM-1]-1]          output_data;
    logic                                     dram_enable;
    logic [pkg_memory::DRAM_ADDR_BITS-1:0]    dram_addr;
    logic [pkg_memory::DRAM_DATA_BITS-1:0]    dram_data;

    int act_file;
    initial begin
        proc_reset = 0;
        proc_start = 0;

        #(CLK_PERIOD) proc_reset = 1;
        #(CLK_PERIOD) proc_reset = 0;

        #(RST_PERIOD);
        act_file = $fopen("bram_activation.mif", "r");
        for (int row = 0; row < 3 * 32; row++) begin
            $fscanf(act_file, "%b", input_data);
            input_en = 1;
            input_addr = row;
            #(CLK_PERIOD);
        end
        input_en = 0;
        $fclose(act_file);

        #(RST_PERIOD) proc_start = 1;
        #(CLK_PERIOD) proc_start = 0;

        wait(proc_finish);
        #(4*CLK_PERIOD) $finish();

    end

    /* Module instantiation */
    network test (
        .*
    );

endmodule


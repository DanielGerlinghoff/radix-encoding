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
    logic proc_reset, proc_start, proc_finish;
    logic dram_enable;
    logic [pkg_memory::DRAM_ADDR_BITS-1:0] dram_addr;
    logic [pkg_memory::DRAM_DATA_BITS-1:0] dram_data;

    initial begin
        proc_reset = 0;
        proc_start = 0;

        #(CLK_PERIOD) proc_reset = 1;
        #(CLK_PERIOD) proc_reset = 0;

        #(RST_PERIOD) proc_start = 1;
        #(CLK_PERIOD) proc_start = 0;

        wait(proc_finish);
        #(4*CLK_PERIOD);
        $finish();

    end

    /* Module instantiation */
    network test (
        .*
    );

endmodule


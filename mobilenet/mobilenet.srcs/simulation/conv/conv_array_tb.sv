`timescale 1ns / 1ps
`default_nettype none
`include "../sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 31/03/2021
//
// Description: Test of design module conv_array
//
//////////////////////////////////////////////////////////////////////////////////


module conv_array_tb;
    import pkg_processing::*;

    /* Clock signal */
    logic clk;
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end

    /* Module input signals */
    localparam ID = 0;
    localparam ID_MEM = 0;
    localparam KER_VALS = KER_SIZE[ID] ** 2;

    logic rst;
    logic start, finish;
    logic activation [CONV_SIZE[ID]];
    logic [CONV_BITS-1:0] row_conv [CONV_SIZE[ID]];

    if_configuration conf ();

    if_kernel ker ();

    initial begin
        rst                 = 0;
        start               = 0;
        activation          = '{default: 0};
        conf.enable[ID]     = 1;
        conf.parallel[ID]   = 2;
        ker.bram_rd_data[ID_MEM] = 0;
        ker.bram_rd_val[ID_MEM]  = 0;

        #(CLK_PERIOD) rst = 1;
        #(CLK_PERIOD) rst = 0;

        #(RST_PERIOD);
        for (int k = 0; k < PARALLEL_MAX[ID]; k++) begin
            #(CLK_PERIOD);
            for (int val = 0; val < KER_VALS * KER_BITS; val++) begin
                ker.bram_rd_data[ID_MEM][val] = $random();
            end
            ker.bram_rd_val[ID_MEM] = 1'b1;

            #(CLK_PERIOD);
            ker.bram_rd_val[ID_MEM] = 1'b0;
        end

        #(RST_PERIOD) start = 1;
        activation = '{default: 1};
        #(CLK_PERIOD) start = 0;

        #(RST_PERIOD);
        $finish;

    end

    /* Module instantiation */
    conv_array #(
        .ID(ID)
    ) test (
        .*
    );

endmodule


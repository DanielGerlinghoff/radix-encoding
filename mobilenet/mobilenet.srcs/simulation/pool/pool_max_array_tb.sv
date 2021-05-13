`timescale 1ns / 1ps
`default_nettype none
`include "../sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 10/05/2021
//
// Description: Test of design module pool_max_array
//
//////////////////////////////////////////////////////////////////////////////////


module pool_max_array_tb;
    import pkg_pooling::*;

    /* Clock signal */
    logic clk;
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end

    /* Module parameters and signals */
    localparam ID = pkg_convolution::CONVUNITS + 0;

    logic rst;
    logic start, finish;
    logic [ACT_BITS-1:0] activation [POOL_SIZE[ID]];
    logic [ACT_BITS-1:0] row_pool [POOL_SIZE[ID]];

    if_configuration conf ();

    initial begin
        rst             = 0;
        start           = 0;
        activation      = '{default: 0};
        conf.enable[ID] = 1;

        #(CLK_PERIOD) rst = 1;
        #(CLK_PERIOD) rst = 0;

        for (int row = 0; row < 2; row++) begin
            #(RST_PERIOD) start = 1;
            #(CLK_PERIOD) start = 0;
            for (int val = 0; val < POOL_SIZE[ID]; val++)
                activation[val] = val + 2 * row;
            #(CLK_PERIOD);
            for (int val = 0; val < POOL_SIZE[ID]; val++)
                activation[val] = val + 2 * row + 1;
        end

        #(RST_PERIOD);
        $finish;

    end

    /* Module instantiation */
    pool_max_array #(
        .ID(ID)
    ) test (
        .*
    );

endmodule


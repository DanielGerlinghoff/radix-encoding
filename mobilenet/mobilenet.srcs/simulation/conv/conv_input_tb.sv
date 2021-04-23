`timescale 1ns / 1ps
`default_nettype none
`include "../sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 05/04/2021
//
// Description: Test of design module conv_input
//
//////////////////////////////////////////////////////////////////////////////////


module conv_input_tb;
    /* Clock signal */
    logic clk;
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end

    /* Module parameters */
    localparam COLS         = 128;
    localparam KERNEL       = 3;
    localparam CONVUNITS    = 1;
    localparam ACT_SIZE_MAX = 224;

    /* Module input signals */
    logic start;

    if_configuration conf ();

    if_activation #(
        .SIZE_MAX(ACT_SIZE_MAX)
    ) act ();

    initial begin
        start    = 0;
        act.data = {ACT_SIZE_MAX/4 {4'h4}};
        act.wren = 0;

        /* Write activation data */
        #(RST_PERIOD);
        act.wren = 1;

        #(CLK_PERIOD);
        act.wren = 0;

        /* Start convolution */
        #(RST_PERIOD);
        conf.parallel[0] = 0;
        conf.stride[0]   = 0;
        start = 1;
        #(CLK_PERIOD);
        start = 0;

        #(RST_PERIOD);
        conf.parallel[0] = 2;
        conf.stride[0]   = 0;
        start = 1;
        #(CLK_PERIOD);
        start = 0;

        #(RST_PERIOD);
        conf.parallel[0] = 4;
        conf.stride[0]   = 0;
        start = 1;
        #(CLK_PERIOD);
        start = 0;

        #(RST_PERIOD);
        conf.parallel[0] = 0;
        conf.stride[0]   = 1;
        start = 1;
        #(CLK_PERIOD);
        start = 0;

        #(RST_PERIOD);
        $finish;
    end

    /* Module instantiation */
    logic act_row [128];

    conv_input #(
        .ID(0),
        .COLS(COLS),
        .KERNEL(KERNEL)
    ) test (
        .*
    );

endmodule


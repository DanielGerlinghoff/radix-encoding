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
    localparam CONVUNITS = 1;
    localparam ID_MEM    = 0;

    /* Module input signals */
    logic start;

    if_configuration conf ();

    if_activation act (.clk);

    initial begin
        start               = 0;
        conf.enable[0]      = 1;
        act.mem_select      = ID_MEM;
        act.rd_data[ID_MEM] = {pkg_memory::ACT_WIDTH_MAX/4 {4'h4}};
        act.rd_val[ID_MEM]  = 0;

        /* Write activation data */
        #(RST_PERIOD) act.rd_val[ID_MEM] = 1;
        #(CLK_PERIOD) act.rd_val[ID_MEM] = 0;

        /* Start convolution */
        #(RST_PERIOD);
        conf.conv_parallel = 0;
        conf.conv_stride   = 0;
        start = 1;
        #(CLK_PERIOD);
        start = 0;

        #(RST_PERIOD);
        conf.conv_parallel = 2;
        conf.conv_stride   = 0;
        start = 1;
        #(CLK_PERIOD);
        start = 0;

        #(RST_PERIOD);
        conf.conv_parallel = 4;
        conf.conv_stride   = 0;
        start = 1;
        #(CLK_PERIOD);
        start = 0;

        #(RST_PERIOD);
        conf.conv_parallel = 0;
        conf.conv_stride   = 1;
        start = 1;
        #(CLK_PERIOD);
        start = 0;

        #(RST_PERIOD);
        $finish;
    end

    /* Module instantiation */
    logic act_row [128];

    conv_input #(
        .ID(0)
    ) test (
        .*
    );

endmodule


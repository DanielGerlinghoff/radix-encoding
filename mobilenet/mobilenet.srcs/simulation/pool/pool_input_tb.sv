`timescale 1ns / 1ps
`default_nettype none
`include "../sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 11/05/2021
//
// Description: Test of design module pool_input
//
//////////////////////////////////////////////////////////////////////////////////


module pool_input_tb;
    import pkg_pooling::*;

    /* Clock signal */
    logic clk;
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end

    /* Module parameters */
    localparam ID     = pkg_convolution::CONVUNITS + 0;
    localparam ID_MEM = 0;

    /* Module input signals */
    logic start;

    if_configuration conf ();
    if_activation act (.clk);

    initial begin
        start               = 0;
        conf.enable[ID]     = 1;
        act.mem_rd_select   = ID_MEM;
        act.rd_data[ID_MEM] = 0;
        act.rd_val[ID_MEM]  = 0;

        /* Write activation data */
        #(RST_PERIOD);
        for (int b = 0; b < ACT_BITS * PARALLEL_MAX[ID]; b++) begin
            #(CLK_PERIOD);
            act.rd_data[ID_MEM] = $random();
            act.rd_val[ID_MEM]  = 1;
            #(CLK_PERIOD);
            act.rd_val[ID_MEM] = 0;
        end

        /* Start convolution */
        #(RST_PERIOD);
        conf.pool_parallel = 0;
        start = 1;
        #(CLK_PERIOD);
        start = 0;

        #(RST_PERIOD);
        conf.pool_parallel = 1;
        start = 1;
        #(CLK_PERIOD);
        start = 0;

        #(RST_PERIOD);
        conf.pool_parallel = 2;
        start = 1;
        #(CLK_PERIOD);
        start = 0;

        #(RST_PERIOD);
        $finish;
    end

    /* Module instantiation */
    logic [ACT_BITS-1:0] act_row [POOL_SIZE[ID]];

    pool_input #(
        .ID(ID)
    ) test (
        .*
    );

endmodule


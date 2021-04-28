`timescale 1ns / 1ps
`default_nettype none
`include "../sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 23/04/2021
//
// Description: Test of design module conv_unit
//
//////////////////////////////////////////////////////////////////////////////////


module conv_unit_tb;
    import pkg_processing::*;

    /* Clock signal */
    logic clk;
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end

    /* Module parameters */
    localparam ID           = 0;
    localparam KER_VALS     = KER_SIZE[ID] ** 2;
    localparam ACT_SIZE_MAX = 224;

    /* Module input signals */
    logic                               rst;
    logic                               start, finish;
    logic [$clog2(CONV_SIZE[ID])-1:0]   act_addr;
    logic [CONV_SIZE[ID]*CONV_BITS-1:0] act_output;

    if_configuration conf ();

    if_kernel #(
        .KER_BITS(KER_BITS),
        .KER_VALS(KER_VALS)
    ) ker ();

    if_activation #(
        .SIZE_MAX(ACT_SIZE_MAX)
    ) act ();


    initial begin
        rst      = 0;
        start    = 0;
        act_addr = 0;

        #(CLK_PERIOD) rst = 1;
        #(CLK_PERIOD) rst = 0;

        /* Configuration */
        #(RST_PERIOD);
        conf.enable[0]   = 1;
        conf.parallel[0] = 0;
        conf.stride[0]   = 0;
        conf.output_mode = conf.DIR;

        /* Load kernel */
        #(RST_PERIOD);
        for (int k = 0; k < PARALLEL_MAX[ID]; k++) begin
            #(CLK_PERIOD);
            for (int val = 0; val < KER_VALS; val++) begin
                ker.data[val] = $random();
            end
            ker.wren = 1'b1;

            #(CLK_PERIOD);
            ker.wren = 1'b0;
        end

        /* Load activation */
        act.data = {ACT_SIZE_MAX/4 {4'h4}};
        #(RST_PERIOD) act.wren = 1;
        #(CLK_PERIOD) act.wren = 0;

        /* Start and reset */
        #(RST_PERIOD) 
        for (int i = 0; i < 4; i++) begin
            start = 1;
            #(CLK_PERIOD);
            start = 0;

            wait(finish);
            #(CLK_PERIOD/2);
        end

        #(RST_PERIOD) rst = 1;
        #(CLK_PERIOD) rst = 0;

        #(CLK_PERIOD);
        $finish;

    end

    /* Module instantiation */
    conv_unit #(
        .ID(ID)
    ) test (
        .*
    );

endmodule


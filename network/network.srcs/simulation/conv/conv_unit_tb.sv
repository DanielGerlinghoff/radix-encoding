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
    import pkg_convolution::*;
    import pkg_memory::*;

    /* Clock signal */
    logic clk;
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end

    /* Module parameters */
    localparam ID       = 0;
    localparam ID_MEM   = 0;
    localparam KER_VALS = KER_SIZE[ID] ** 2;

    /* Module input signals */
    if_configuration conf ();
    if_control ctrl ();
    if_kernel ker (.clk);
    if_activation act (.clk);

    initial begin
        ctrl.reset = 0;
        ctrl.start = 0;

        #(CLK_PERIOD) ctrl.reset = 1;
        #(CLK_PERIOD) ctrl.reset = 0;

        /* Configuration */
        #(RST_PERIOD);
        conf.enable[ID]    = 1;
        conf.conv_parallel = 0;
        conf.conv_stride   = 0;
        conf.conv_padding  = 0;
        conf.output_mode   = conf.DIR;
        act.mem_rd_select  = ID_MEM;
        act.mem_wr_select  = ID_MEM + 1;
        ker.ker_select     = ID_MEM;

        /* Load kernel */
        #(RST_PERIOD);
        for (int k = 0; k < PARALLEL_MAX[ID]; k++) begin
            #(CLK_PERIOD);
            for (int val = 0; val < KER_VALS * KER_BITS; val++) begin
                ker.ker_bram_rd_data[ID_MEM][val] = $random();
            end
            ker.ker_bram_rd_val[ID_MEM] = 1'b1;

            #(CLK_PERIOD);
            ker.ker_bram_rd_val[ID_MEM] = 1'b0;
        end

        /* Load activation */
        act.rd_data[ID_MEM] = {ACT_WIDTH_MAX/4 {4'h4}};
        #(RST_PERIOD) act.rd_val[ID_MEM] = 1;
        #(CLK_PERIOD) act.rd_val[ID_MEM] = 0;

        /* Start and reset */
        #(RST_PERIOD);
        for (int i = 0; i < 4; i++) begin
            ctrl.start = 1;
            #(CLK_PERIOD);
            ctrl.start = 0;

            wait(ctrl.finish[ID]);
            #(CLK_PERIOD/2);
        end

        #(RST_PERIOD) ctrl.reset = 1;
        #(CLK_PERIOD) ctrl.reset = 0;

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


`timescale 1ns / 1ps
`default_nettype none
`include "../sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 12/05/2021
//
// Description: Test of design module pool_max_unit
//
//////////////////////////////////////////////////////////////////////////////////


module pool_max_unit_tb;
    import pkg_pooling::*;
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
    localparam ID     = pkg_convolution::CONVUNITS + 0;
    localparam ID_MEM = 0;
    localparam PAR    = 1;

    /* Module input signals */
    if_configuration conf ();
    if_control ctrl ();
    if_activation act (.clk);

    initial begin
        ctrl.reset = 0;
        ctrl.start = 0;

        #(CLK_PERIOD) ctrl.reset = 1;
        #(CLK_PERIOD) ctrl.reset = 0;

        /* Configuration */
        #(RST_PERIOD);
        conf.enable[ID]    = 1;
        conf.pool_parallel = PAR;
        act.mem_select     = ID_MEM;

        /* Start and reset */
        #(RST_PERIOD);
        for (int i = 0; i < KER_SIZE[ID]; i++) begin
            for (int a = 0; a < PARALLEL_NUM[ID][PAR]; a++) begin
                for (int b = 0; b < ACT_BITS; b++) begin
                    #(CLK_PERIOD);
                    act.rd_data[ID_MEM] = $random();
                    act.rd_val[ID_MEM] = 1;
                    #(CLK_PERIOD);
                    act.rd_val[ID_MEM] = 0;
                end
            end

            #(CLK_PERIOD) ctrl.start = 1;
            #(CLK_PERIOD) ctrl.start = 0;

            wait(ctrl.finish[ID]);
            #(CLK_PERIOD/2);
        end

        #(RST_PERIOD) ctrl.reset = 1;
        #(CLK_PERIOD) ctrl.reset = 0;

        #(CLK_PERIOD);
        $finish;

    end

    /* Module instantiation */
    pool_max_unit #(
        .ID(ID)
    ) test (
        .*
    );

endmodule


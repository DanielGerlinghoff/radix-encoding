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
    logic                               rst;
    logic                               start, finish;
    logic [$clog2(CONV_SIZE[ID])-1:0]   act_addr;
    logic [CONV_SIZE[ID]*CONV_BITS-1:0] act_output;

    if_configuration conf ();

    if_kernel ker (.clk);

    if_activation act (.clk);

    initial begin
        rst      = 0;
        start    = 0;
        act_addr = 0;

        #(CLK_PERIOD) rst = 1;
        #(CLK_PERIOD) rst = 0;

        /* Configuration */
        #(RST_PERIOD);
        conf.enable[0]     = 1;
        conf.mem_select    = ID_MEM;
        conf.conv_parallel = 0;
        conf.conv_stride   = 0;
        conf.output_mode   = conf.DIR;

        /* Load kernel */
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

        /* Load activation */
        act.rd_data[ID_MEM] = {ACT_WIDTH_MAX/4 {4'h4}};
        #(RST_PERIOD) act.rd_val[ID_MEM] = 1;
        #(CLK_PERIOD) act.rd_val[ID_MEM] = 0;

        /* Start and reset */
        #(RST_PERIOD);
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


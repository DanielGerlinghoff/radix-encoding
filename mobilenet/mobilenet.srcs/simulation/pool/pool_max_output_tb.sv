`timescale 1ns / 1ps
`default_nettype none
`include "../sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 12/05/2021
//
// Description: Test of design modules pool_max_output
//
//////////////////////////////////////////////////////////////////////////////////


module pool_max_output_tb;
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
    localparam ID     = 0;
    localparam ID_MEM = 0;

    /* Module input signals */
    if_configuration conf ();
    if_activation act (.clk);

    logic                rst;
    logic                pool_valid;
    logic [ACT_BITS-1:0] pool_row [POOL_SIZE[ID]];

    initial begin
        rst        = 0;
        pool_valid = 0;

        conf.pool_enable   = '{1};
        conf.pool_parallel = 1;
        act.mem_select     = ID_MEM;
        act.addr_step      = '{8, 12};

        #(CLK_PERIOD) rst = 1;
        #(CLK_PERIOD) rst = 0;

        #(RST_PERIOD);
        for (int r = 0; r < 5; r++) begin
            #(8*CLK_PERIOD);
            pool_row   = '{default: r};
            pool_valid = 1;

            #(CLK_PERIOD);
            pool_valid = 0;
        end

        #(RST_PERIOD);
        $finish;
    end

    /* Module instantiation */
    pool_max_output #(
        .ID(ID)
    ) test (
        .*
    );

endmodule


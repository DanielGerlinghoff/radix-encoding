`timescale 1ns / 1ps
`default_nettype none
`include "../sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 02/05/2021
//
// Description: Test of design module conv_relu
//
//////////////////////////////////////////////////////////////////////////////////


module conv_relu_tb;
    import pkg_convolution::*;

    /* Clock signal */
    logic clk;
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end

    /* Module parameters */
    localparam ID  = 0;
    localparam PAR = 2;

    /* Module input signals */
    logic [0:CONV_SIZE[ID]*CONV_BITS-1] conv_data;

    if_configuration conf ();
    if_activation act (.clk);

    generate
        for (genvar a = 0; a < PARALLEL_NUM[ID][PAR]; a++) begin
            localparam low  = PARALLEL_OUT[ID][PAR][a][0] * CONV_BITS;
            localparam high = (PARALLEL_OUT[ID][PAR][a][1] + 1) * CONV_BITS - 1;
            initial conv_data[low:high] = $random();
        end
    endgenerate

    initial begin
        conf.conv_parallel = PAR;
        conf.act_scale     = 4;
        act.mem_wr_select  = 1;
        act.wr_addr_base   = 0;
        act.conv_rd_en[ID] = 0;
        act.addr_step      = '{4, 10};

        #(RST_PERIOD) act.conv_rd_en[ID] = 1;
        #(CLK_PERIOD) act.conv_rd_en[ID] = 0;

        wait(|act.transfer_finish);
        #(4*CLK_PERIOD);
        $finish();

    end

    /* Module instantiation */
    conv_relu #(
        .ID(ID)
    ) test (
        .*
    );

endmodule


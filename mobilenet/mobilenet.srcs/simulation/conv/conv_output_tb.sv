`timescale 1ns / 1ps
`default_nettype none
`include "../sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 22/04/2021
//
// Description: Test of design modules conv_output and conv_bram
//
//////////////////////////////////////////////////////////////////////////////////


module conv_output_tb;
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
    localparam ID = 0;

    /* Module input signals */
    if_configuration conf ();

    logic                 rst;
    logic                 act_valid;
    logic [CONV_BITS-1:0] act_row [CONV_SIZE[ID]];

    initial begin
        rst       = 0;
        act_row   = '{default: 1};
        act_valid = 0;

        conf.enable[ID]       = 1;
        conf.conv_output_mode = conf.DIR;

        #(CLK_PERIOD);
        rst = 1;
        #(CLK_PERIOD);
        rst = 0;

        for (int o = 0; o < 3; o++) begin
            #(RST_PERIOD);
            for (int r = 0; r < 5; r++) begin
                #(3*CLK_PERIOD);
                act_row   = '{default: r};
                act_valid = 1;

                #(CLK_PERIOD);
                act_valid = 0;
            end

            #(CLK_PERIOD);
            conf.conv_output_mode = conf.conv_output_mode.next();
            rst = 1;
            #(CLK_PERIOD);
            rst = 0;

        end

        #(RST_PERIOD);
        $finish;
    end

    /* Module instantiation */
    wire [$clog2(CONV_SIZE[ID])-1:0]   bram_addr;
    wire [CONV_SIZE[ID]*CONV_BITS-1:0] bram_rd_data;
    wire [CONV_SIZE[ID]*CONV_BITS-1:0] bram_wr_data;
    wire                               bram_wr_en;

    conv_output #(
        .ID(ID)
    ) test (
        .*
    );

    conv_bram #(
        .WIDTH (CONV_SIZE[ID]*CONV_BITS),
        .HEIGHT(CONV_SIZE[ID])
    ) mem (
        .clk       (clk),
        .en_a      (conf.enable[ID]),
        .addr_a    (bram_addr),
        .rd_data_a (bram_rd_data),
        .wr_en_a   (bram_wr_en),
        .wr_data_a (bram_wr_data),
        .rd_en_b   (0),
        .rd_addr_b (0),
        .rd_data_b ()
    );

endmodule


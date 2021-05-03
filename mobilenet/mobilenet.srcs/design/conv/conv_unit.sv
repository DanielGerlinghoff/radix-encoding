`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 23/04/2021
//
// Description: Combine input, convolution array, output logic and BRAM
//
//////////////////////////////////////////////////////////////////////////////////


module conv_unit
import pkg_processing::*;
#(
    ID
) (
    if_configuration conf,
    if_kernel ker,
    if_activation act,
    input  logic clk, rst,
    input  logic start,
    output logic finish
);

    /* Convolution modules */
    wire                               conv_input [CONV_SIZE[ID]];
    wire [CONV_BITS-1:0]               conv_output [CONV_SIZE[ID]];
    wire [CONV_SIZE[ID]*CONV_BITS-1:0] conv_result;
    wire                               conv_finish;

    wire [$clog2(CONV_SIZE[ID])-1:0]   bram_addr;
    wire [CONV_SIZE[ID]*CONV_BITS-1:0] bram_rd_data;
    wire [CONV_SIZE[ID]*CONV_BITS-1:0] bram_wr_data;
    wire                               bram_wr_en;

    conv_input #(
        .ID(ID)
    ) inp (
        .conf    (conf),
        .act     (act),
        .clk     (clk),
        .start   (start),
        .act_row (conv_input)
    );

    conv_array #(
        .ID(ID)
    ) array (
        .ker        (ker),
        .clk        (clk),
        .rst        (rst),
        .start      (start),
        .finish     (conv_finish),
        .activation (conv_input),
        .row_conv   (conv_output)
    );

    conv_output #(
        .ID(ID)
    ) out (
        .conf      (conf),
        .clk       (clk),
        .rst       (rst),
        .act_row   (conv_output),
        .act_valid (conv_finish),


        .bram_addr,
        .bram_rd_data,
        .bram_wr_data,
        .bram_wr_en
    );

    conv_bram #(
        .WIDTH  (CONV_SIZE[ID]*CONV_BITS),
        .HEIGHT (CONV_SIZE[ID])
    ) bram (
        .clk       (clk),
        .en_a      (conf.enable[ID]),
        .addr_a    (bram_addr),
        .rd_data_a (bram_rd_data),
        .wr_en_a   (bram_wr_en),
        .wr_data_a (bram_wr_data),
        .rd_en_b   (act.conv_rd_en[ID]),
        .rd_addr_b (act.conv_rd_addr),
        .rd_data_b (conv_result)
    );

    conv_relu #(
        .ID(ID)
    ) relu (
        .clk       (clk),
        .conf      (conf),
        .act       (act),
        .conv_data (conv_result)
    );

    assign finish = conv_finish;

endmodule


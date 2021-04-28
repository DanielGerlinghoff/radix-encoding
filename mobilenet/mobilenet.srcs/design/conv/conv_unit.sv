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
    output logic finish,

    input  logic [$clog2(CONV_SIZE[ID])-1:0]   act_addr,
    output logic [CONV_SIZE[ID]*CONV_BITS-1:0] act_output
);

    /* Convolution modules */
    wire                 conv_input [CONV_SIZE[ID]];
    wire [CONV_BITS-1:0] conv_output [CONV_SIZE[ID]];
    wire                 conv_finish;

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
    ) mem (
        .clk       (clk),
        .enable    (conf.enable[ID]),
        .addr_a    (bram_addr),
        .addr_b    (act_addr),
        .wr_data   (bram_wr_data),
        .wr_en     (bram_wr_en),
        .rd_data_a (bram_rd_data),
        .rd_data_b (act_output)
    );

    assign finish = conv_finish;

endmodule


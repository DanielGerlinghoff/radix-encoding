`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 05/04/2021
//
// Description: Connect activation BRAMs to convolution arrays
//
//////////////////////////////////////////////////////////////////////////////////


interface if_activation (
    input logic clk
);

    import pkg_memory::*;

    logic [$clog2(ACT_NUM)-1:0]        mem_rd_select, mem_wr_select;
    logic [$clog2(ACT_HEIGHT_MAX)-1:0] addr_step [2];
    tri0                               transfer_finish;

    /* Activation BRAMs */
    logic [$clog2(ACT_HEIGHT_MAX)-1:0] wr_addr;
    logic [$clog2(ACT_HEIGHT_MAX)-1:0] wr_addr_base;
    tri0  [$clog2(ACT_HEIGHT_MAX)-1:0] wr_addr_offset;
    tri0                               wr_add_addr;
    tri0                               wr_en [ACT_NUM];
    tri0  [0:ACT_WIDTH_MAX-1]          wr_data;
    logic                              rd_en [ACT_NUM-1];
    logic [$clog2(ACT_HEIGHT_MAX)-1:0] rd_addr;
    logic                              rd_val [ACT_NUM];
    logic [0:ACT_WIDTH_MAX-1]          rd_data [ACT_NUM];

    always_ff @(posedge clk) begin
        for (int n = 0; n < ACT_NUM; n++) begin
            rd_val[n] <= rd_en[n];
        end

        if (wr_add_addr) begin
            wr_addr <= wr_addr_base + wr_addr_offset;
        end
    end

    /* Input and output access */
    logic                                     in_en, in_en_dly;
    logic [$clog2(ACT_HEIGHT_MAX)-1:0]        in_addr, in_addr_dly;
    logic [0:ACT_WIDTH_MAX-1]                 in_data, in_data_dly;
    logic                                     out_en;
    logic [$clog2(ACT_HEIGHT[ACT_NUM-1])-1:0] out_addr;
    logic [0:ACT_WIDTH[ACT_NUM-1]-1]          out_data;

    assign wr_addr_offset = in_en     ? in_addr_dly : 'z;
    assign wr_add_addr    = in_en     ? 1 : 'z;
    assign wr_en[0]       = in_en_dly ? 1 : 'z;
    assign wr_data        = in_en_dly ? in_data_dly : 'z;

    always @(posedge clk) begin
        in_en_dly   <= in_en;
        in_addr_dly <= in_addr;
        in_data_dly <= in_data;
    end

    /* Intermediate convolution BRAM */
    import pkg_convolution::CONVUNITS;
    import pkg_convolution::CONV_SIZE_MAX;

    logic                              conv_wr_en [CONVUNITS];
    logic                              conv_rd_en [CONVUNITS];
    logic                              conv_rd_val [CONVUNITS];
    logic [$clog2(CONV_SIZE_MAX)-1:0]  conv_rd_addr;

    always_ff @(posedge clk) begin
        for (int n = 0; n < CONVUNITS; n++) begin
            conv_rd_val[n] <= conv_rd_en[n];
        end
    end

    /* Modports */
    modport proc(
        output mem_rd_select,
        output mem_wr_select,
        output addr_step,
        input  transfer_finish,
        output rd_en,
        output rd_addr,
        output wr_addr_base,
        output conv_rd_en,
        output conv_rd_addr,
        input  conv_wr_en
    );

    modport conv_in (
        input mem_rd_select,
        input rd_data,
        input rd_val
    );

    modport conv_relu (
        input  mem_wr_select,
        input  addr_step,
        output transfer_finish,
        input  conv_rd_val,
        output wr_addr_offset,
        output wr_add_addr,
        output wr_en,
        output wr_data
    );

    modport pool_in (
        input mem_rd_select,
        input rd_data,
        input rd_val
    );

    modport pool_out (
        input  mem_wr_select,
        input  addr_step,
        output transfer_finish,
        output wr_addr_offset,
        output wr_add_addr,
        output wr_en,
        output wr_data
    );

    modport lin (
        input  mem_rd_select,
        input  mem_wr_select,
        input  addr_step,
        output transfer_finish,
        input  rd_data,
        input  rd_val,
        output wr_addr_offset,
        output wr_add_addr,
        output wr_en,
        output wr_data
    );

    modport bram (
        input  wr_en,
        input  wr_addr,
        input  wr_data,
        input  rd_en,
        input  rd_addr,
        output rd_data,

        input  in_en,
        input  in_addr,
        input  in_data,
        input  out_en,
        input  out_addr,
        output out_data
    );

endinterface


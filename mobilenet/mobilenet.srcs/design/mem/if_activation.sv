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
    import pkg_processing::CONVUNITS;
    import pkg_processing::CONV_SIZE_MAX;
    import pkg_processing::CONV_BITS;

    logic [$clog2(ACT_HEIGHT_MAX)-1:0] addr;
    logic [$clog2(ACT_HEIGHT_MAX)-1:0] wr_addr_base;
    logic [$clog2(ACT_HEIGHT_MAX)-1:0] wr_addr_offset;
    logic                              wr_add_addr;
    logic                              wr_en [ACT_NUM];
    logic [ACT_WIDTH_MAX-1:0]          wr_data;
    logic                              rd_en [ACT_NUM];
    logic                              rd_val [ACT_NUM];
    logic [ACT_WIDTH_MAX-1:0]          rd_data [ACT_NUM];

    always_ff @(posedge clk) begin
        for (int n = 0; n < ACT_NUM; n++) begin
            rd_val[n] <= rd_en[n];
        end

        if (wr_add_addr) begin
            addr <= wr_addr_base + wr_addr_offset;
        end
    end

    logic                              conv_rd_en [CONVUNITS];
    logic                              conv_rd_val [CONVUNITS];
    logic [$clog2(CONV_SIZE_MAX)-1:0]  conv_rd_addr;
    logic [$clog2(CONV_BITS)-1:0]      conv_scale;
    logic [$clog2(ACT_HEIGHT_MAX)-1:0] conv_addr_step [2];
    logic                              conv_transfer_finish = 0;

    always_ff @(posedge clk) begin
        for (int n = 0; n < CONVUNITS; n++) begin
            conv_rd_val[n] <= conv_rd_en[n];
        end
    end

    /* Modports */
    modport array_in (
        input rd_data,
        input rd_val
    );

    modport bram (
        input  addr,
        input  wr_en,
        input  wr_data,
        input  rd_en,
        output rd_data
    );

    modport array_relu (
        input  conv_rd_val,
        input  conv_scale,
        input  conv_addr_step,
        output conv_transfer_finish,
        output wr_addr_offset,
        output wr_add_addr,
        output wr_en,
        output wr_data
    );

endinterface


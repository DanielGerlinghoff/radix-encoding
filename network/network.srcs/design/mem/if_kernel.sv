`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 01/04/2021
//
// Description: Connect kernel BRAMs to convolution arrays
//
//////////////////////////////////////////////////////////////////////////////////


interface if_kernel (
    input logic clk
);

    import pkg_memory::*;

    /* Convolution kernel BRAM */
    logic [$clog2(KER_NUM+1)-1:0] ker_select;

    logic                                 ker_bram_wr_en [KER_NUM];
    logic [DRAM_ADDR_BITS-1:0]            ker_bram_wr_addr;
    logic [DRAM_DATA_BITS-1:0]            ker_bram_wr_data;
    logic                                 ker_bram_rd_en [KER_NUM];
    logic                                 ker_bram_rd_val [KER_NUM];
    logic [$clog2(KER_HEIGHT_MAX[0])-1:0] ker_bram_rd_addr;
    logic [KER_WIDTH_MAX-1:0]             ker_bram_rd_data [KER_NUM];

    always_ff @(posedge clk) begin
        for (int n = 0; n < KER_NUM; n++) begin
            ker_bram_rd_val[n] <= ker_bram_rd_en[n];
        end
    end

    /* Linear weights BRAM */
    logic [$clog2(WGT_NUM+1)-1:0] wgt_select;

    logic                              wgt_bram_wr_en [WGT_NUM];
    logic [$clog2(WGT_HEIGHT_MAX)-1:0] wgt_bram_wr_addr;
    logic [DRAM_DATA_BITS-1:0]         wgt_bram_wr_data;
    logic                              wgt_bram_rd_en [WGT_NUM];
    logic                              wgt_bram_rd_val [WGT_NUM];
    logic [$clog2(WGT_HEIGHT_MAX)-1:0] wgt_bram_rd_addr;
    logic [DRAM_DATA_BITS-1:0]         wgt_bram_rd_data [WGT_NUM];

    always_ff @(posedge clk) begin
        for (int n = 0; n < KER_NUM; n++) begin
            wgt_bram_rd_val[n] <= wgt_bram_rd_en[n];
        end
    end

    /* Modports */
    modport proc (
        output ker_select,
        output wgt_select,
        output ker_bram_rd_en,
        output ker_bram_rd_addr,
        output wgt_bram_rd_en,
        output wgt_bram_rd_addr
    );

    modport conv (
        input ker_select,
        input ker_bram_rd_data,
        input ker_bram_rd_val
    );

    modport lin (
        input wgt_select,
        input wgt_bram_rd_data,
        input wgt_bram_rd_val
    );

    modport bram (
        input  ker_bram_wr_en,
        input  ker_bram_wr_addr,
        input  ker_bram_wr_data,
        input  ker_bram_rd_en,
        input  ker_bram_rd_addr,
        output ker_bram_rd_data,

        input  wgt_bram_wr_en,
        input  wgt_bram_wr_addr,
        input  wgt_bram_wr_data,
        input  wgt_bram_rd_en,
        input  wgt_bram_rd_addr,
        output wgt_bram_rd_data
    );

endinterface


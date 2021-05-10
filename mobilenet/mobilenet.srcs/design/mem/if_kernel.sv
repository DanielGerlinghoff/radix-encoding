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

    logic [$clog2(KER_NUM)-1:0] mem_select;

    logic                                 bram_wr_en [KER_NUM];
    logic [DRAM_ADDR_BITS-1:0]            bram_wr_addr;
    logic [DRAM_DATA_BITS-1:0]            bram_wr_data;
    logic                                 bram_rd_en [KER_NUM];
    logic                                 bram_rd_val [KER_NUM];
    logic [$clog2(KER_HEIGHT_MAX[0])-1:0] bram_rd_addr;
    logic [KER_WIDTH_MAX-1:0]             bram_rd_data [KER_NUM];

    always_ff @(posedge clk) begin
        for (int n = 0; n < KER_NUM; n++) begin
            bram_rd_val[n] <= bram_rd_en[n];
        end
    end

    /* Modports */
    modport proc (
        output mem_select,
        output bram_rd_en,
        output bram_rd_addr
    );

    modport array (
        input mem_select,
        input bram_rd_data,
        input bram_rd_val
    );

    modport bram (
        input  bram_wr_en,
        input  bram_wr_addr,
        input  bram_wr_data,
        input  bram_rd_en,
        input  bram_rd_addr,
        output bram_rd_data
    );


endinterface


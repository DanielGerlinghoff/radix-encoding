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

    logic ker_n_wgt;

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
        for (int n = 0; n < WGT_NUM; n++) begin
            wgt_bram_rd_val[n] <= wgt_bram_rd_en[n];
        end
    end

    /* External DRAM */
    logic                                  dram_start;
    logic [pkg_memory::DRAM_ADDR_BITS-1:0] dram_cnt;
    logic                                  dram_en = 0, dram_val [2];
    logic [pkg_memory::DRAM_ADDR_BITS-1:0] dram_addr_base, dram_addr;
    logic [pkg_memory::DRAM_DATA_BITS-1:0] dram_data;
    logic                                  dram_rdy;

    always_ff @(posedge clk) begin
        if (dram_start) begin
            dram_addr <= dram_addr_base;
            dram_en   <= 1;
        end
        if (dram_en && dram_rdy) begin
            if (dram_cnt != 1) begin
                dram_cnt  <= dram_cnt - 1;
                dram_addr <= dram_addr + 1;
            end else begin
                dram_cnt  <= 0;
                dram_addr <= 0;
                dram_en   <= 0;
            end
        end
    end

    always_ff @(posedge clk) begin
        dram_val[1] <= dram_val[0];

        if (dram_start) begin
            ker_bram_wr_addr <= 0;
            wgt_bram_wr_addr <= 0;
        end else if (dram_val[0]) begin
            if (ker_n_wgt) begin
                ker_bram_wr_data           <= dram_data;
                ker_bram_wr_en[ker_select] <= 1;
            end else begin
                wgt_bram_wr_data           <= dram_data;
                wgt_bram_wr_en[wgt_select] <= 1;
            end
        end else begin
            ker_bram_wr_en[ker_select] <= 0;
            wgt_bram_wr_en[wgt_select] <= 0;
        end

        if (dram_val[1]) begin
            if (ker_n_wgt) begin
                ker_bram_wr_addr <= ker_bram_wr_addr + 1;
            end else begin
                wgt_bram_wr_addr <= wgt_bram_wr_addr + 1;
            end
        end
    end

endinterface


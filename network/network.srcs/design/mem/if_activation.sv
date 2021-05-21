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
    import pkg_convolution::CONVUNITS, pkg_pooling::POOLUNITS, pkg_linear::LINUNITS;

    localparam UNITS = CONVUNITS + POOLUNITS + LINUNITS;

    logic [$clog2(ACT_NUM)-1:0]        mem_rd_select, mem_wr_select;
    logic [$clog2(ACT_HEIGHT_MAX)-1:0] addr_step [2];
    logic                              transfer_finish [UNITS];

    /* Input and output access */
    logic                                     in_en, in_en_dly;
    logic [$clog2(ACT_HEIGHT_MAX)-1:0]        in_addr, in_addr_dly;
    logic [0:ACT_WIDTH_MAX-1]                 in_data, in_data_dly;
    logic                                     out_en;
    logic [$clog2(ACT_HEIGHT[ACT_NUM-1])-1:0] out_addr;
    logic [ACT_WIDTH[ACT_NUM-1]-1:0]          out_data;

    /* Activation BRAMs */
    logic [$clog2(ACT_HEIGHT_MAX)-1:0] wr_addr;
    logic [$clog2(ACT_HEIGHT_MAX)-1:0] wr_addr_base;
    logic [$clog2(ACT_HEIGHT_MAX)-1:0] wr_addr_offset_u [UNITS];
    logic [0:ACT_NUM-1]                wr_en, wr_en_u [UNITS];
    logic [0:ACT_WIDTH_MAX-1]          wr_data, wr_data_u [UNITS];
    logic [0:ACT_NUM-2]                rd_en;
    logic [$clog2(ACT_HEIGHT_MAX)-1:0] rd_addr;
    logic                              rd_val [ACT_NUM];
    logic [0:ACT_WIDTH_MAX-1]          rd_data [ACT_NUM];

    always_ff @(posedge clk) begin
        for (int n = 0; n < ACT_NUM - 1; n++) begin
            rd_val[n] <= rd_en[n];
        end

        for (int u = 0; u < UNITS; u++) begin
            if (|wr_en_u[u]) begin
                wr_en   <= wr_en_u[u];
                wr_addr <= wr_addr_base + wr_addr_offset_u[u];
                wr_data <= wr_data_u[u];
            end
        end
        if (in_en) begin
            wr_en[0] <= 1;
            wr_addr  <= in_addr;
            wr_data  <= in_data;
        end
        if (!(in_en || |(UNITS*ACT_NUM)'({>>{wr_en_u}}))) begin
            wr_en <= 0;
        end
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

endinterface


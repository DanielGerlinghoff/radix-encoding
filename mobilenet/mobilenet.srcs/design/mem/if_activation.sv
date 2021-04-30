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

    logic [$clog2(ACT_HEIGHT_MAX)-1:0] addr;
    logic                              wr_en [ACT_NUM];
    logic [ACT_WIDTH_MAX-1:0]          wr_data;
    logic                              rd_en [ACT_NUM];
    logic                              rd_val [ACT_NUM];
    logic [ACT_WIDTH_MAX-1:0]          rd_data [ACT_NUM];

    always_ff @(posedge clk) begin
        for (int n = 0; n < ACT_NUM; n++) begin
            rd_val[n] <= rd_en[n];
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

endinterface


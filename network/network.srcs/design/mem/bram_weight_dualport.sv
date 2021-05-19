`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 18/05/2021
//
// Description: Dual port BRAM writing data from DRAM and reading it to linear
//              units
//
//////////////////////////////////////////////////////////////////////////////////


module bram_weight_dualport #(
    WIDTH,
    HEIGHT
) (
    input  logic                      clk,
    input  logic                      wr_en,
    input  logic [$clog2(HEIGHT)-1:0] wr_addr,
    input  logic [WIDTH-1:0]          wr_data,
    input  logic                      rd_en,
    input  logic [$clog2(HEIGHT)-1:0] rd_addr,
    output logic [WIDTH-1:0]          rd_data
);

    (* rom_style = "block" *) logic [WIDTH-1:0] ram [HEIGHT];

    always_ff @(posedge clk) begin
        if (rd_en) begin
            rd_data <= ram[rd_addr];
        end
    end

    always_ff @(posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
    end

endmodule


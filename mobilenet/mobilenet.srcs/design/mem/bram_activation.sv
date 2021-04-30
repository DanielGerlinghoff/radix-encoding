`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 30/04/2021
//
// Description: Simple dual port BRAM to store activations
//
//////////////////////////////////////////////////////////////////////////////////


module bram_activation #(
    WIDTH,
    HEIGHT
) (
    input  logic                      clk,
    input  logic [$clog2(HEIGHT)-1:0] addr,
    input  logic                      wr_en,
    input  logic [WIDTH-1:0]          wr_data,
    input  logic                      rd_en,
    output logic [WIDTH-1:0]          rd_data
);

    logic [WIDTH-1:0] ram [HEIGHT];

    always_ff @(posedge clk) begin
        if (rd_en) begin
            rd_data <= ram[addr];
        end

        if (wr_en) begin
            ram[addr] <= wr_data;
        end
    end

endmodule


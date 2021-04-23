`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 22/04/2021
//
// Description: BRAM to store intermediate convolution values
//
//////////////////////////////////////////////////////////////////////////////////


module conv_bram #(
    WIDTH,
    HEIGHT
) (
    input  logic                      clk,
    input  logic                      enable,
    input  logic [$clog2(HEIGHT)-1:0] addr_a,
    input  logic [$clog2(HEIGHT)-1:0] addr_b,
    input  logic [WIDTH-1:0]          wr_data,
    input  logic                      wr_en,
    output logic [WIDTH-1:0]          rd_data_a,
    output logic [WIDTH-1:0]          rd_data_b
);

    logic [WIDTH-1:0] ram [HEIGHT-1:0];

    always_ff @(posedge clk) begin
        if (enable) begin
            rd_data_a <= ram[addr_a];
            rd_data_b <= ram[addr_b];

            if (wr_en) begin
                ram[addr_a] <= wr_data;
            end
        end
    end

endmodule


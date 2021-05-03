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
    input  logic                      en_a,
    input  logic [$clog2(HEIGHT)-1:0] addr_a,
    output logic [WIDTH-1:0]          rd_data_a,
    input  logic                      wr_en_a,
    input  logic [WIDTH-1:0]          wr_data_a,
    input  logic                      rd_en_b,
    input  logic [$clog2(HEIGHT)-1:0] rd_addr_b,
    output logic [WIDTH-1:0]          rd_data_b
);

    logic [WIDTH-1:0] ram [HEIGHT-1:0];

    always_ff @(posedge clk) begin
        if (en_a) begin
            rd_data_a <= ram[addr_a];

            if (wr_en_a) begin
                ram[addr_a] <= wr_data_a;
            end
        end

        if (rd_en_b) begin
            rd_data_b <= ram[rd_addr_b];
        end
    end

endmodule


`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 29/03/2021
//
// Description: Adds zero or addend to accumulator based on select value
//              Clears accumulator on clear signal
//
//////////////////////////////////////////////////////////////////////////////////


module conv_accumulator #(
    INP_SIZE,
    OUT_SIZE
) (
    input  logic                       clk,
    input  logic signed [INP_SIZE-1:0] addend,
    input  logic                       select,
    input  logic                       clear,
    input  logic                       acc_wren,
    input  logic signed [OUT_SIZE-1:0] acc_in,
    output logic signed [OUT_SIZE-1:0] acc_out
);

    always_ff @(posedge clk) begin
        if (clear) begin
            acc_out <= 0;
        end else if (acc_wren) begin
            acc_out <= acc_in;
        end else if (select) begin
            acc_out <= acc_out + addend;
        end
    end

endmodule


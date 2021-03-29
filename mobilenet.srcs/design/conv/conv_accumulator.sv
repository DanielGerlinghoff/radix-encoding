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
    input  logic                clk,
    input  logic [INP_SIZE-1:0] addend,
    input  logic                select,
    input  logic                clear,
    output logic [OUT_SIZE-1:0] accumulator  
);

    always_ff @(posedge clk) begin
        if (clear) begin
            accumulator <= 0;
        end else begin
            if (select) begin
                accumulator <= accumulator + addend;
            end
        end
    end

endmodule


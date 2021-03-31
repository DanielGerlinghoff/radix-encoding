`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 29/03/2021
// 
// Description: Combines accumulators to calculate one row of activations
// 
//////////////////////////////////////////////////////////////////////////////////


module conv_row #(
    COLS,
    KER_SIZE,
    SUM_SIZE
) (
    input  logic                clk,
    input  logic                clear,
    input  logic                enable,
    input  logic                activation [COLS],
    input  logic [KER_SIZE-1:0] kernel [COLS],
    output logic [SUM_SIZE-1:0] sum [COLS]
);

    for (genvar col = 0; col < COLS; col++) begin :conv_accs
        conv_accumulator #(
            .INP_SIZE(KER_SIZE),
            .OUT_SIZE(SUM_SIZE)
        ) acc_i (
            .clk         (clk),
            .addend      (kernel[col]),
            .select      (activation[col] & enable),
            .clear       (clear),
            .accumulator (sum[col])
        );
    end

endmodule


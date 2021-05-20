`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 12/05/2021
//
// Description: Combine input, pooling array and output logic
//
//////////////////////////////////////////////////////////////////////////////////


module pool_max_unit
import pkg_pooling::*;
#(
    ID
) (
    input logic clk,
    if_configuration conf,
    if_control ctrl,
    if_activation act
);

    /* Pooling modules */
    logic                pool_start;
    wire                 pool_finish;
    wire  [ACT_BITS-1:0] pool_input [POOL_SIZE[ID]];
    wire  [ACT_BITS-1:0] pool_output [POOL_SIZE[ID]];

    always_ff @(posedge clk) begin
        pool_start <= ctrl.start;
    end

    pool_input #(
        .ID(ID)
    ) inp (
        .conf    (conf),
        .act     (act),
        .clk     (clk),
        .start   (ctrl.start),
        .act_row (pool_input)
    );

    pool_max_array #(
        .ID(ID)
    ) array (
        .conf       (conf),
        .clk        (clk),
        .rst        (ctrl.reset),
        .start      (pool_start),
        .finish     (pool_finish),
        .activation (pool_input),
        .row_pool   (pool_output)
    );

    pool_max_output #(
        .ID(ID)
    ) out (
        .clk        (clk),
        .conf       (conf),
        .act        (act),
        .pool_row   (pool_output),
        .pool_valid (pool_finish)
    );

    assign ctrl.finish[ID] = pool_finish;

endmodule


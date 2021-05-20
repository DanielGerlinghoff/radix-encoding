`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 10/05/2021
//
// Description: Maximum pooling operation with equal kernel size and stride and
//              kernel size as a power-of-two
//
//////////////////////////////////////////////////////////////////////////////////


module pool_max_array
import pkg_pooling::*;
#(
    ID
) (
    if_configuration conf,
    input  logic                clk, rst,
    input  logic                start,
    output logic                finish,
    input  logic [ACT_BITS-1:0] activation [POOL_SIZE[ID]],
    output logic [ACT_BITS-1:0] row_pool [POOL_SIZE[ID]]
);

    /* Process control */
    enum logic [1:0] {
        IDLE  = 2'b01,
        POOL  = 2'b10
    } state = IDLE;
    logic [$clog2(KER_SIZE[ID])-1:0] col_cnt;

    always_ff @(posedge clk) begin
        unique case (state)
            IDLE: begin
                finish <= 0;
                if (conf.enable[ID] && start) begin
                    col_cnt <= 0;
                    state <= POOL;
                end
            end

            POOL: begin
                if (col_cnt != KER_SIZE[ID] - 1) begin
                    col_cnt <= col_cnt + 1;
                end else begin
                    col_cnt <= 0;
                    finish <= 1;
                    state <= IDLE;
                end
            end
        endcase
    end

    /* Maximum function */
    function logic [ACT_BITS-1:0] max (input logic [ACT_BITS-1:0] val_0, val_1);
        if (val_0 >= val_1) max = val_0;
        else                max = val_1;
    endfunction

    always_ff @(posedge clk) begin
        if (rst) begin
            row_pool <= '{default: 0};
        end else if (state == POOL) begin
            for (int c = 0; c < POOL_SIZE[ID]; c++) begin
                row_pool[c] <= max(activation[c], row_pool[c]);
            end
        end
    end

endmodule


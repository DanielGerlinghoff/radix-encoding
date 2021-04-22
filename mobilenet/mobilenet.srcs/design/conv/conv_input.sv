`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 05/04/2021
//
// Description: Prepare and shift activations for convolution array
//
//////////////////////////////////////////////////////////////////////////////////


module conv_input #(
    ID,
    COLS,
    KERNEL
) (
    if_activation.array_in act,
    if_configuration.array_in conf,
    input  logic clk,
    input  logic start,
    output logic act_row [COLS]
);

    import pkg_configuration::*;

    localparam KERNEL_HALF = (KERNEL - 1) / 2;
    localparam COLS_INPUT  = STRIDE_MAX * COLS + (KERNEL - 1);

    /* Activation register */
    logic [act.SIZE_MAX-1:0] act_reg;

    always_ff @(posedge clk) begin
        if (act.wren) begin
            act_reg <= act.data;
        end
    end

    /* Parallel assignment */
    wor [COLS_INPUT-1:0] act_parallel = 0;

    generate
        for (genvar p = 0; p < PARALLEL_DIM[0]; p++) begin :gen_parallel
            for (genvar a = 0; a < PARALLEL_NUM[p]; a++) begin :gen_parallel_assign
                assign act_parallel[PARALLEL[p][a][1]:PARALLEL[p][a][0]] = (conf.parallel[ID] == p) ? act_reg : 'z;
            end
        end
    endgenerate

    /* Row shift */
    logic [COLS_INPUT-1:0] act_shift;
    logic [$clog2(KERNEL)-1:0] shift_cnt;

    always_ff @(posedge clk) begin
        if (start) begin
            act_shift <= act_parallel << KERNEL_HALF;
            shift_cnt <= 1;
        end else if (shift_cnt > 0 && shift_cnt < KERNEL) begin
            act_shift <= act_shift >> 1;
            shift_cnt <= shift_cnt + 1;
        end else begin
            shift_cnt <= 0;
        end
    end

    /* Stride selection */
    wire [COLS-1:0] act_stride;

    generate
        for (genvar s = 0; s < STRIDE_DIM; s++) begin :gen_stride
            for (genvar i = 0; i < COLS; i++) begin :gen_stride_assign
                assign act_stride[i] = (conf.stride[ID] == s) ? act_shift[i*STRIDE[s]+KERNEL_HALF] : 'z;
            end
        end
    endgenerate

    /* Register output */
    always_ff @(posedge clk) begin
        if (shift_cnt) begin
            for (int i = 0; i < COLS; i++) begin
                act_row[i] <= act_stride[i];
            end
        end
    end

endmodule


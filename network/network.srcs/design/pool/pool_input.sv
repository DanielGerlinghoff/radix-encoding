`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 11/05/2021
//
// Description: Prepare and shift activations for pooling array
//
//////////////////////////////////////////////////////////////////////////////////


module pool_input
import pkg_pooling::*;
#(
    ID
) (
    if_configuration conf,
    if_activation act,
    input  logic                clk,
    input  logic                start,
    output logic [ACT_BITS-1:0] act_row [POOL_SIZE[ID]]
);

    localparam SIZE       = POOL_SIZE[ID];
    localparam SIZE_INPUT = KER_SIZE[ID] * SIZE;

    /* Activation register */
    import pkg_memory::ACT_WIDTH_MAX;
    logic [0:ACT_WIDTH_MAX-1][ACT_BITS-1:0] act_reg [PARALLEL_MAX[ID]];
    logic [$clog2(PARALLEL_MAX[ID])-1:0]    assign_cnt = 0;
    logic [$clog2(ACT_BITS)-1:0]            bit_cnt = 0;

    always_ff @(posedge clk) begin
        if (conf.enable[ID] && act.rd_val[act.mem_rd_select]) begin
            for (int val = 0; val < ACT_WIDTH_MAX; val++) begin
                act_reg[assign_cnt][val][ACT_BITS-bit_cnt-1] <= act.rd_data[act.mem_rd_select][val];
            end

            if (bit_cnt < ACT_BITS - 1) begin
                bit_cnt <= bit_cnt + 1;
            end else begin
                bit_cnt <= 0;
                if (assign_cnt < PARALLEL_NUM[ID][conf.pool_parallel] - 1) begin
                    assign_cnt <= assign_cnt + 1;
                end else begin
                    assign_cnt <= 0;
                end
            end
        end
    end

    /* Parallel assignment */
    logic [0:PARALLEL_DIM[ID][0]-1][0:SIZE_INPUT-1][ACT_BITS-1:0] act_parallel_p;
    logic [0:SIZE_INPUT-1][ACT_BITS-1:0] act_parallel;

    generate
        for (genvar p = 0; p < PARALLEL_DIM[ID][0]; p++) begin :gen_parallel
            for (genvar a = 0; a < PARALLEL_NUM[ID][p]; a++) begin :gen_parallel_assign
                localparam int pos [2] = PARALLEL_IN[ID][p][a];
                assign act_parallel_p[p][pos[0]:pos[1]] = act_reg[a][0:pos[1]-pos[0]];
            end
        end
    endgenerate

    assign act_parallel = act_parallel_p[conf.pool_parallel];

    /* Row shift */
    logic [0:SIZE_INPUT-1][ACT_BITS-1:0] act_shift;
    logic [$clog2(KER_SIZE[ID]+1)-1:0] shift_cnt;

    always_ff @(posedge clk) begin
        if (conf.enable[ID] && start) begin
            act_shift <= act_parallel;
            shift_cnt <= 1;
        end else if (shift_cnt > 0 && shift_cnt < KER_SIZE[ID]) begin
            act_shift[SIZE_INPUT-1] <= 0;
            for (int val = 0; val < SIZE_INPUT - 1; val++)
                act_shift[val] <= act_shift[val+1];
            shift_cnt <= shift_cnt + 1;
        end else begin
            shift_cnt <= 0;
        end
    end

    /* Stride selection */
    wire [0:SIZE-1][ACT_BITS-1:0] act_stride;

    generate
        for (genvar a = 0; a < SIZE; a++) begin :gen_stride_assign
            assign act_stride[a] = act_shift[a*KER_SIZE[ID]];
        end
    endgenerate

    /* Register output */
    always_ff @(posedge clk) begin
        if (shift_cnt) begin
            {>>{act_row}} <= act_stride;
        end
    end

endmodule


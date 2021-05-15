`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 05/04/2021
//
// Description: Prepare and shift activations for convolution array
//
//////////////////////////////////////////////////////////////////////////////////


module conv_input
import pkg_convolution::*;
#(
    ID
) (
    if_configuration.conv_in conf,
    if_activation.conv_in act,
    input  logic clk,
    input  logic start,
    output logic act_row [CONV_SIZE[ID]]
);

    localparam SIZE          = CONV_SIZE[ID];
    localparam KER_SIZE_HALF = (KER_SIZE[ID] - 1) / 2;
    localparam SIZE_INPUT    = STRIDE_MAX[ID] * SIZE + (KER_SIZE[ID] - 1);

    /* Activation register */
    import pkg_memory::ACT_WIDTH_MAX;
    logic [0:ACT_WIDTH_MAX-1] act_reg;

    always_ff @(posedge clk) begin
        if (conf.enable[ID] && act.rd_val[act.mem_select]) begin
            act_reg <= act.rd_data[act.mem_select];
        end
    end

    /* Parallel assignment */
    tri0 [0:SIZE_INPUT+KER_SIZE_HALF-1] act_parallel;

    generate
        for (genvar p = 0; p < PARALLEL_DIM[ID][0]; p++) begin :gen_parallel
            for (genvar a = 0; a < PARALLEL_NUM[ID][p]; a++) begin :gen_parallel_assign
                localparam int pos [2] = PARALLEL_IN[ID][p][a];
                assign act_parallel[pos[0]:pos[1]] = (conf.conv_parallel == p) ? act_reg[0:pos[1]-pos[0]] : 'z;
            end
        end
    endgenerate

    /* Row shift */
    logic [0:SIZE_INPUT+KER_SIZE_HALF-1] act_shift;
    logic [$clog2(KER_SIZE[ID]+1)-1:0] shift_cnt;

    always_ff @(posedge clk) begin
        if (conf.enable[ID] && start) begin
            act_shift <= act_parallel >> KER_SIZE_HALF;
            shift_cnt <= 1;
        end else if (shift_cnt > 0 && shift_cnt < KER_SIZE[ID]) begin
            act_shift <= act_shift << 1;
            shift_cnt <= shift_cnt + 1;
        end else begin
            shift_cnt <= 0;
        end
    end

    /* Stride selection */
    wire [0:SIZE-1] act_stride;

    generate
        for (genvar s = 0; s < STRIDE_DIM[ID]; s++) begin :gen_stride
            for (genvar i = 0; i < SIZE; i++) begin :gen_stride_assign
                assign act_stride[i] = (conf.conv_stride == s) ? act_shift[i*STRIDE[ID][s]+KER_SIZE_HALF] : 'z;
            end
        end
    endgenerate

    /* Register output */
    always_ff @(posedge clk) begin
        if (shift_cnt) begin
            {>>{act_row}} <= act_stride;
        end
    end

endmodule


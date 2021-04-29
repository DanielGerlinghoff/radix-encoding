`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 31/03/2021
//
// Description: Convolution array processing three rows in parallel
//
//////////////////////////////////////////////////////////////////////////////////


module conv_array
import pkg_processing::*;
#(
    ID
) (
    if_kernel.array ker,
    input  logic clk, rst,
    input  logic start,
    output logic finish,
    input  logic activation [CONV_SIZE[ID]],
    output logic [CONV_BITS-1:0] row_conv [CONV_SIZE[ID]]
);

    /* Kernel registers */
    import pkg_mapping::KER_TO_CONV;

    localparam KER_REGS = PARALLEL_MAX[ID];
    localparam KER_VALS = KER_SIZE[ID] ** 2;
    localparam ID_MEM   = KER_TO_CONV[ID];

    logic [KER_VALS-1:0][KER_BITS-1:0] kernel_regs [KER_REGS];
    logic [$clog2(KER_REGS+1)-1:0] kernel_cnt = 0;

    always_ff @(posedge clk) begin
        if (rst) begin
            kernel_cnt <= 0;
        end else if (conf.enable[ID] && ker.bram_rd_val[ID_MEM]) begin
            kernel_regs[kernel_cnt] <= ker.bram_rd_data[ID_MEM][KER_VALS*KER_BITS-1:0];
            kernel_cnt <= kernel_cnt + 1;
        end
    end

    /* Process control */
    enum logic [3:0] {
        IDLE   = 4'b0001,
        KERNEL = 4'b0010,
        CONV   = 4'b0100,
        SHIFT  = 4'b1000
    } state = IDLE;
    logic [$clog2(KER_SIZE[ID])-1:0] col_cnt;
    logic conv_clear, conv_enable;

    always_ff @(posedge clk) begin
        unique case (state)
            IDLE: begin
                if (conf.enable[ID] && start) begin
                    col_cnt <= 0;
                    state <= KERNEL;
                end
            end

            KERNEL: begin
                col_cnt <= col_cnt + 1;
                state <= CONV;
            end

            CONV: begin
                if (col_cnt != KER_SIZE[ID]) begin
                    col_cnt <= col_cnt + 1;
                end else begin
                    col_cnt <= 0;
                    state <= SHIFT;
                end
            end

            SHIFT: begin
                state <= IDLE;
            end

        endcase
    end

    assign conv_clear  = rst;
    assign conv_enable = (state == CONV);

    /* Kernel assignment */
    tri0  [KER_BITS-1:0] kernel_parallel [KER_SIZE[ID]][CONV_SIZE[ID]];
    logic [KER_BITS-1:0] kernel_rows [KER_SIZE[ID]][CONV_SIZE[ID]];

    generate
        for (genvar r = 0; r < KER_SIZE[ID]; r++) begin :gen_parallel_rows
            for (genvar p = 0; p < PARALLEL_DIM[ID][0]; p++) begin :gen_parallel
                for (genvar a = 0; a < PARALLEL_NUM[ID][p]; a++) begin :gen_parallel_assign
                    for (genvar v = PARALLEL_KER[ID][p][a][0]; v <= PARALLEL_KER[ID][p][a][1]; v++) begin :gen_parallel_values
                        assign kernel_parallel[r][v] = (conf.conv_parallel == p) ? kernel_regs[a][col_cnt+r*KER_SIZE[ID]] : 'z;
                    end
                end
            end
        end
    endgenerate

    always_ff @(posedge clk) begin
        if ((state & (KERNEL | CONV)) && col_cnt < KER_SIZE[ID]) begin
            kernel_rows <= kernel_parallel;
        end
    end

    /* Sum shift */
    logic [CONV_BITS-1:0] sum_in [KER_SIZE[ID]][CONV_SIZE[ID]], sum_out [KER_SIZE[ID]][CONV_SIZE[ID]];
    logic sum_wren;

    assign sum_wren = (state == SHIFT);

    always_comb begin
        sum_in[0] = '{default: 0};
        for (int row = 0; row < KER_SIZE[ID] - 1; row++) begin
            sum_in[row+1] = sum_out[row];
        end
    end

    always_ff @(posedge clk) begin
        if (state & SHIFT) begin
            row_conv <= sum_out[KER_SIZE[ID]-1];
            finish <= 1;
        end else begin
            finish <= 0;
        end
    end

    /* Instantiate rows */
    for (genvar row = 0; row < KER_SIZE[ID]; row++) begin :conv_rows
        conv_row #(
            .COLS(CONV_SIZE[ID]),
            .KER_SIZE(KER_BITS),
            .SUM_SIZE(CONV_BITS)
        ) row_i (
            .clk        (clk),
            .enable     (conv_enable),
            .clear      (conv_clear),
            .activation (activation),
            .kernel     (kernel_rows[row]),
            .sum_wren   (sum_wren),
            .sum_in     (sum_in[row]),
            .sum_out    (sum_out[row])
        );
    end

endmodule


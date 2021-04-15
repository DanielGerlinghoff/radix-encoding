`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 31/03/2021
// 
// Description: Convolution array processing three rows in parallel
// 
//////////////////////////////////////////////////////////////////////////////////


module conv_array #(
    ROWS,
    COLS,
    KER_SIZE,
    KER_VALS,
    SUM_SIZE
) (
    if_kernel.array ker,
    input  logic clk,
    input  logic start, stop,
    input  logic activation [ROWS][COLS],
    output logic [SUM_SIZE-1:0] row_conv [COLS]  // TODO Include in interface
);

    /* Kernel registers */
    localparam KER_REGS = 16;

    logic [KER_VALS-1:0][KER_SIZE-1:0] kernel_regs [KER_REGS];
    logic [$clog2(KER_REGS)-1:0] kernel_cnt = 0;

    always_ff @(posedge clk) begin
        if (ker.wren) begin
            kernel_regs[kernel_cnt] <= ker.data;
            kernel_cnt <= kernel_cnt + 1;
        end

        if (stop) begin
            kernel_cnt <= 0;
        end
    end

    /* Process control */
    enum logic [3:0] {
        IDLE   = 4'b0001,
        KERNEL = 4'b0010,
        CONV   = 4'b0100,
        SHIFT  = 4'b1000
    } state = IDLE;
    logic [1:0] col_cnt;
    logic conv_enable, conv_clear;
        
    always_ff @(posedge clk) begin
        conv_enable <= 0;
        conv_clear  <= 0;

        unique case (state)
            IDLE: begin
                if (start) begin
                    col_cnt <= 0;
                    conv_clear <= 1;

                    state <= KERNEL;
                end
            end

            KERNEL: begin
                col_cnt <= col_cnt + 1;
                conv_enable <= 1;

                state <= CONV;
            end

            CONV: begin
                if (col_cnt != 3) begin
                    col_cnt <= col_cnt + 1;
                    conv_enable <= 1;
                end else begin
                    col_cnt <= 0;
                    if (!stop) begin
                        state <= KERNEL;
                    end else begin
                        state <= SHIFT;
                    end
                end
            end

            SHIFT: begin
                state <= IDLE;
            end

        endcase
    end

    /* Kernel assignment */
    logic [KER_SIZE-1:0] kernel_rows [ROWS][COLS];

    always_ff @(posedge clk) begin
        if (state & (KERNEL | CONV)) begin
            unique case (kernel_cnt)
                1: begin
                    for (int col = 0; col < COLS; col++) begin
                        kernel_rows[0][col] <= kernel_regs[0][0+col_cnt];
                        kernel_rows[1][col] <= kernel_regs[0][3+col_cnt];
                        kernel_rows[2][col] <= kernel_regs[0][6+col_cnt];
                    end
                end

                // TODO Other kernel_cnt
            endcase
        end
    end

    /* Sum shift */
    logic [SUM_SIZE-1:0] sum_in [ROWS][COLS], sum_out [ROWS][COLS];
    logic sum_wren;

    assign sum_wren = (state == KERNEL);

    always_comb begin
        sum_in[0] = '{default: 0};
        for (int row = 0; row < ROWS - 1; row++) begin
            sum_in[row+1] = sum_out[row];
        end
    end

    always_ff @(posedge clk) begin
        if (state & (KERNEL | SHIFT)) begin
            row_conv <= sum_out[ROWS-1];
        end
    end
    
    /* Instantiate rows */
    for (genvar row = 0; row < ROWS; row++) begin :conv_rows
        conv_row #(
            .COLS(COLS),
            .KER_SIZE(KER_SIZE),
            .SUM_SIZE(SUM_SIZE)
        ) row_i (
            .clk        (clk),
            .enable     (conv_enable),
            .clear      (conv_clear),
            .activation (activation[row]),
            .kernel     (kernel_rows[row]),
            .sum_wren   (sum_wren),
            .sum_in     (sum_in[row]),
            .sum_out    (sum_out[row])
        );
    end

endmodule


`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 18/05/2021
//
// Description: Top level block with device inputs and outputs
//
//////////////////////////////////////////////////////////////////////////////////


module top (
    input  wire clk_user_p,
    input  wire clk_user_n,
    input  wire sys_clk_p,
    input  wire sys_clk_n,

    inout  wire uart_rst_n,
    input  wire uart_txd,
    input  wire uart_rts,
    output wire uart_rxd,
    output wire uart_cts
);

    import pkg_memory::*;
    import pkg_linear::SUM_BITS, pkg_linear::CHANNELS_OUT;

    /* Input and clock buffer */
    localparam CLKPERIOD = 5e-9;
    localparam CLKDIV    = 1;

    wire         clk_200, clk_100, clk;
    logic [31:0] clk_cnt [2];

    IBUFDS inp_buf (
        .I  (sys_clk_p),
        .IB (sys_clk_n),
        .O  (clk_200)
    );

    BUFGCE_DIV #(
        .BUFGCE_DIVIDE (CLKDIV)
    ) clk_buf (
        .I   (clk_200),
        .O   (clk_100),
        .CE  (1'b1),
        .CLR (1'b0)
    );

    assign clk = clk_100;

    /* UART controller */
    localparam UART_BITS   = 8;
    localparam INPUT_SIZE  = ACT_WIDTH_MAX;
    localparam OUTPUT_SIZE = ACT_HEIGHT[ACT_NUM-1];

    logic [UART_BITS-1:0] uart_tx_data;
    logic                 uart_tx_en;
    logic                 uart_tx_rdy_n;
    logic [UART_BITS-1:0] uart_rx_data;
    logic                 uart_rx_valid;

    logic                              reset, start, finish;
    logic                              input_en;
    logic [$clog2(ACT_HEIGHT_MAX)-1:0] input_addr;
    logic [0:ACT_WIDTH_MAX-1]          input_data;
    logic [$clog2(INPUT_SIZE+1)-1:0]   input_cnt;
    logic                              output_en, output_val;
    logic [$clog2(CHANNELS_OUT+1)-1:0] output_addr;
    logic [SUM_BITS-1:0]               output_data;

    enum logic [4:0] {
        IDLE   = 5'b00001,
        RESET  = 5'b00010,
        START  = 5'b00100,
        INPUT  = 5'b01000,
        OUTPUT = 5'b10000
    } state = IDLE;

    uart #(
        .CLKPERIOD (CLKPERIOD * CLKDIV),
        .STOPBITS  (2)
    ) uart_ctrl (
        .clk      (clk),
        .rstn     (uart_rst_n),
        .rxd      (uart_txd),
        .cts      (uart_rts),
        .txd      (uart_rxd),
        .rts      (uart_cts),

        .tx_data  (uart_tx_data),
        .tx_en    (uart_tx_en),
        .tx_rdy_n (uart_tx_rdy_n),
        .rx_data  (uart_rx_data),
        .rx_valid (uart_rx_valid)
    );

    logic iterate, count;

    always_ff @(posedge clk) begin
        reset      <= 0;
        start      <= 0;
        input_en   <= 0;
        output_en  <= 0;
        uart_tx_en <= 0;

        case (state)
            IDLE: begin
                if (uart_rx_valid) begin
                    if (uart_rx_data == "R") state   <= RESET;
                    if (uart_rx_data == "S") state   <= START;
                    if (uart_rx_data == "I") state   <= INPUT;
                    if (uart_rx_data == "T") iterate <= 1;
                    if (uart_rx_data == "C") count   <= 1;
                end
                if (finish) state <= OUTPUT;
            end

            RESET: begin
                reset       <= 1;
                input_cnt   <= 0;
                input_addr  <= 0;
                output_addr <= 0;
                iterate     <= 0;
                count       <= 0;
                state <= IDLE;
            end

            START: begin
                start <= 1;
                state <= IDLE;
            end

            INPUT: begin
                if (uart_rx_valid) begin
                    input_cnt <= input_cnt + UART_BITS;
                    input_data[input_cnt+:UART_BITS] <= uart_rx_data;
                end
                if (input_cnt >= INPUT_SIZE) begin
                    input_cnt <= 0;
                    input_en  <= 1;
                end
                if (input_en) begin
                    if (input_addr != pkg_convolution::ACT_BITS * INPUT_SIZE - 1) begin
                        input_addr <= input_addr + 1;
                    end else begin
                        input_addr <= 0;
                        state <= IDLE;
                    end
                end
            end

            OUTPUT: begin
                if (iterate) begin
                    start <= 1;
                    state <= IDLE;

                end else if (count) begin
                    uart_tx_en   <= 1;
                    uart_tx_data <= clk_cnt[1][output_addr*UART_BITS+:UART_BITS];
                    if (output_addr != 3) begin
                        output_addr <= output_addr + 1;
                    end else begin
                        output_addr <= 0;
                        state <= IDLE;
                    end
                end else begin
                    output_en <= !uart_tx_rdy_n;
                    output_val <= output_en;
                    if (output_en) begin
                        if (output_addr != CHANNELS_OUT) begin
                            output_addr <= output_addr + 1;
                        end else begin
                            output_addr <= 0;
                            state <= IDLE;
                        end
                    end
                    uart_tx_en   <= output_val;
                    uart_tx_data <= output_data[UART_BITS-1:0];
                end
            end
        endcase
    end

    always_ff @(posedge clk) begin
        if (state & RESET || finish) begin
            clk_cnt[0] <= 0;
            clk_cnt[1] <= clk_cnt[0];
        end else if (state & START || clk_cnt[0] != 0) begin
            clk_cnt[0] <= clk_cnt[0] + 1;
        end
    end

    /* Network accelerator */
    network net (
        .clk         (clk),
        .proc_reset  (reset),
        .proc_start  (start),
        .proc_finish (finish),
        .input_en,
        .input_addr,
        .input_data,
        .output_en,
        .output_addr,
        .output_data,
        .dram_enable (),
        .dram_addr   (),
        .dram_data   (0)
    );

endmodule


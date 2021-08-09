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
    input  wire        clk_user_p,
    input  wire        clk_user_n,
    input  wire        sys_clk_p,
    input  wire        sys_clk_n,

    input  wire        uart_txd,
    input  wire        uart_rts,
    output wire        uart_rxd,
    output wire        uart_cts,

    output wire        c0_ddr4_act_n,
    output wire [16:0] c0_ddr4_adr,
    output wire [1:0]  c0_ddr4_ba,
    output wire [1:0]  c0_ddr4_bg,
    output wire [0:0]  c0_ddr4_cke,
    output wire [0:0]  c0_ddr4_odt,
    output wire [0:0]  c0_ddr4_cs_n,
    output wire [0:0]  c0_ddr4_ck_t,
    output wire [0:0]  c0_ddr4_ck_c,
    output wire        c0_ddr4_reset_n,
    inout  wire [7:0]  c0_ddr4_dm_dbi_n,
    inout  wire [63:0] c0_ddr4_dq,
    inout  wire [7:0]  c0_ddr4_dqs_t,
    inout  wire [7:0]  c0_ddr4_dqs_c
);

    import pkg_memory::*;
    import pkg_linear::SUM_BITS, pkg_linear::CHANNELS_OUT;

    /* Input and clock buffer */
    localparam CLKPERIOD = DRAM_ENABLE ? 4.284e-9 : 4.998e-9;  // = 1 / MMCM_CLKOUT : 1 / CLKREF
    localparam CLKDIV    = 1;

    wire         clk_ibuf, clk_buf, clk_ui, clk;
    logic [31:0] clk_cnt [2];
    logic        reset = 0;

    generate
        if (!DRAM_ENABLE) begin
            IBUFDS inp_buffer (
                .I  (sys_clk_p),
                .IB (sys_clk_n),
                .O  (clk_ibuf)
            );

            BUFGCE_DIV #(
                .BUFGCE_DIVIDE (CLKDIV)
            ) clk_buffer (
                .I   (clk_ibuf),
                .O   (clk_buf),
                .CE  (1'b1),
                .CLR (1'b0)
            );

            assign clk = clk_buf;

        end else begin
            assign clk = clk_ui;

        end
    endgenerate

    /* DRAM controller */
    localparam DRAM_MASK_BITS  = DRAM_DATA_BITS / 8;
    localparam DRAM_ADDR_SHIFT = 3;

    logic                      ddr4_app_rdy;
    logic [DRAM_ADDR_BITS-1:0] ddr4_app_addr;
    logic [2:0]                ddr4_app_cmd;
    logic                      ddr4_app_en;
    logic                      ddr4_app_wdf_rdy;
    logic                      ddr4_app_wdf_en;
    logic [DRAM_ADDR_BITS-1:0] ddr4_app_wdf_addr;
    logic [DRAM_DATA_BITS-1:0] ddr4_app_wdf_data;
    logic                      ddr4_app_wdf_end;
    logic [DRAM_MASK_BITS-1:0] ddr4_app_wdf_mask;
    logic                      ddr4_app_rd_en;
    logic [DRAM_ADDR_BITS-1:0] ddr4_app_rd_addr;
    logic [DRAM_DATA_BITS-1:0] ddr4_app_rd_data;
    logic                      ddr4_app_rd_data_valid;
    logic                      ddr4_app_rd_data_end;
    logic                      ddr4_init_calib_complete;

    always_comb begin
        if (ddr4_app_wdf_en) begin
            ddr4_app_en      = 1;
            ddr4_app_cmd     = 0;
            ddr4_app_addr    = ddr4_app_wdf_addr;
            ddr4_app_wdf_end = 1;
        end else if (ddr4_app_rd_en) begin
            ddr4_app_en      = 1;
            ddr4_app_cmd     = 1;
            ddr4_app_addr    = ddr4_app_rd_addr;
            ddr4_app_wdf_end = 0;
        end else begin
            ddr4_app_en      = 0;
            ddr4_app_cmd     = 0;
            ddr4_app_addr    = 0;
            ddr4_app_wdf_end = 0;
        end
        ddr4_app_wdf_mask = 0;
    end

    generate
        if (DRAM_ENABLE) begin
            ddr4_4g_x64 ddr4_control (
                .sys_rst                   (reset),
                .c0_sys_clk_p              (sys_clk_p),
                .c0_sys_clk_n              (sys_clk_n),

                .c0_ddr4_act_n             (c0_ddr4_act_n),
                .c0_ddr4_adr               (c0_ddr4_adr),
                .c0_ddr4_ba                (c0_ddr4_ba),
                .c0_ddr4_bg                (c0_ddr4_bg),
                .c0_ddr4_cke               (c0_ddr4_cke),
                .c0_ddr4_odt               (c0_ddr4_odt),
                .c0_ddr4_cs_n              (c0_ddr4_cs_n),
                .c0_ddr4_ck_t              (c0_ddr4_ck_t),
                .c0_ddr4_ck_c              (c0_ddr4_ck_c),
                .c0_ddr4_reset_n           (c0_ddr4_reset_n),
                .c0_ddr4_dm_dbi_n          (c0_ddr4_dm_dbi_n),
                .c0_ddr4_dq                (c0_ddr4_dq),
                .c0_ddr4_dqs_c             (c0_ddr4_dqs_c),
                .c0_ddr4_dqs_t             (c0_ddr4_dqs_t),

                .c0_ddr4_ui_clk            (clk_ui),
                .c0_ddr4_ui_clk_sync_rst   (),
                .c0_ddr4_app_rdy           (ddr4_app_rdy),
                .c0_ddr4_app_addr          (ddr4_app_addr << DRAM_ADDR_SHIFT),
                .c0_ddr4_app_cmd           (ddr4_app_cmd),
                .c0_ddr4_app_en            (ddr4_app_en),
                .c0_ddr4_app_hi_pri        (1'b0),
                .c0_ddr4_app_wdf_rdy       (ddr4_app_wdf_rdy),
                .c0_ddr4_app_wdf_data      (ddr4_app_wdf_data),
                .c0_ddr4_app_wdf_end       (ddr4_app_wdf_end),
                .c0_ddr4_app_wdf_mask      (ddr4_app_wdf_mask),
                .c0_ddr4_app_wdf_wren      (ddr4_app_wdf_en & ddr4_app_rdy),
                .c0_ddr4_app_rd_data       (ddr4_app_rd_data),
                .c0_ddr4_app_rd_data_end   (ddr4_app_rd_data_end),
                .c0_ddr4_app_rd_data_valid (ddr4_app_rd_data_valid),
                .c0_init_calib_complete    (ddr4_init_calib_complete),

                .dbg_clk                   (),
                .dbg_bus                   ()
            );

        end else begin
            assign {c0_ddr4_act_n, c0_ddr4_adr, c0_ddr4_ba, c0_ddr4_bg, c0_ddr4_odt, c0_ddr4_cs_n, c0_ddr4_reset_n} = '1;
            assign {c0_ddr4_dm_dbi_n, c0_ddr4_dq, c0_ddr4_dqs_t, c0_ddr4_dqs_c} = 'z;
            assign {c0_ddr4_cke, ddr4_app_rdy, ddr4_app_rd_data, ddr4_app_rd_data_valid} = '0;

            OBUFDS OBUFDS_inst (
                .I  (0),
                .O  (c0_ddr4_ck_t),
                .OB (c0_ddr4_ck_c)
            );
        end
    endgenerate

`ifdef DEBUG
    ila_ddr4 ila (
        .clk,
        .probe0 (ddr4_app_addr[9:0]),
        .probe1 (ddr4_app_cmd),
        .probe2 (ddr4_app_en),
        .probe3 (ddr4_app_wdf_data[255:0]),
        .probe4 (ddr4_app_rd_data[255:0]),
        .probe5 (ddr4_app_rd_data_valid),
        .probe6 (ddr4_app_rdy),
        .probe7 (ddr4_app_wdf_rdy),
        .probe8 (state)
    );
`endif

    /* UART controller */
    localparam UART_BITS   = 8;
    localparam INPUT_SIZE  = ACT_WIDTH_MAX;
    localparam OUTPUT_SIZE = ACT_HEIGHT[ACT_NUM-1];
    localparam RST_PERIOD  = 4096;

    logic [UART_BITS-1:0] uart_tx_data;
    logic                 uart_tx_en;
    logic                 uart_tx_rdy_n;
    logic [UART_BITS-1:0] uart_rx_data;
    logic                 uart_rx_valid;

    logic                                start, finish;
    logic [$clog2(RST_PERIOD)-1:0]       reset_cnt;
    logic                                input_en;
    logic [$clog2(ACT_HEIGHT_MAX)-1:0]   input_addr;
    logic [0:ACT_WIDTH_MAX-1]            input_data;
    logic [$clog2(DRAM_DATA_BITS+1)-1:0] input_cnt;
    logic                                output_en, output_val;
    logic [$clog2(CHANNELS_OUT+1)-1:0]   output_addr;
    logic [SUM_BITS-1:0]                 output_data;

    enum logic [5:0] {
        IDLE   = 6'b000001,
        RESET  = 6'b000010,
        START  = 6'b000100,
        INPUT  = 6'b001000,
        OUTPUT = 6'b010000,
        DRAM   = 6'b100000
    } state = IDLE;

    logic count;
    always_ff @(posedge clk) begin
        start           <= 0;
        input_en        <= 0;
        output_en       <= 0;
        output_val      <= 0;
        uart_tx_en      <= 0;
        ddr4_app_wdf_en <= 0;

        case (state)
            IDLE: begin
                if (uart_rx_valid) begin
                    if (uart_rx_data == "R") state <= RESET;
                    if (uart_rx_data == "S") state <= START;
                    if (uart_rx_data == "I") state <= INPUT;
                    if (uart_rx_data == "D") state <= DRAM;
                    if (uart_rx_data == "C") count <= 1;
                end
                if (finish) state <= OUTPUT;
            end

            RESET: begin
                input_cnt         <= 0;
                input_addr        <= 0;
                output_addr       <= 0;
                ddr4_app_wdf_addr <= 0;
                count             <= 0;

                if (!reset) begin
                    reset     <= 1;
                    reset_cnt <= 0;
                end else begin
                    if (reset_cnt != RST_PERIOD - 2) begin
                        reset_cnt <= reset_cnt + 1;
                    end else begin
                        reset_cnt <= 0;
                        reset     <= 0;
                        state <= IDLE;
                    end
                end
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
                if (count) begin
                    uart_tx_en   <= 1;
                    uart_tx_data <= clk_cnt[1][output_addr*UART_BITS+:UART_BITS];
                    if (output_addr != 3) begin
                        output_addr <= output_addr + 1;
                    end else begin
                        output_addr <= 0;
                        state <= IDLE;
                    end
                end else begin
                    output_en  <= !uart_tx_rdy_n;
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

            DRAM: begin
                if (uart_rx_valid) begin
                    input_cnt <= input_cnt + UART_BITS;
                    ddr4_app_wdf_data[DRAM_DATA_BITS-input_cnt-1-:UART_BITS] <= uart_rx_data;
                end
                if (input_cnt >= DRAM_DATA_BITS) begin
                    input_cnt <= 0;
                    ddr4_app_wdf_en <= 1;
                end
                if (ddr4_app_wdf_en) begin
                    if (!ddr4_app_rdy) begin
                        ddr4_app_wdf_en <= 1;
                    end else begin
                        if (ddr4_app_wdf_addr != DRAM_HEIGHT - 1) begin
                            ddr4_app_wdf_addr <= ddr4_app_wdf_addr + 1;
                        end else begin
                            ddr4_app_wdf_addr <= 0;
                            state <= IDLE;
                        end
                    end
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

    uart_ctrl uart_control (
        .clk      (clk),
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
        .dram_rdy  (ddr4_app_rdy),
        .dram_en   (ddr4_app_rd_en),
        .dram_addr (ddr4_app_rd_addr),
        .dram_data (ddr4_app_rd_data),
        .dram_val  (ddr4_app_rd_data_valid)
    );

endmodule

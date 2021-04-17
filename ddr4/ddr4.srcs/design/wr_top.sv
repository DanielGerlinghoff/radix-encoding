`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 16/04/2021
// 
// Description: Top module to test write and read on DRAM hardware
// 
//////////////////////////////////////////////////////////////////////////////////


module wr_top (
    input  wire        sys_rst,

    input  wire        c0_sys_clk_p,
    input  wire        c0_sys_clk_n,
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
    inout  wire [7:0]  c0_ddr4_dqs_c,

    inout  wire        uart_rst_n,
    input  wire        uart_txd,
    input  wire        uart_rts,
    output wire        uart_rxd,
    output wire        uart_cts
);

    /* Test write and read */
    localparam APP_ADDR_WIDTH = 29;
    localparam APP_DATA_WIDTH = 512;
    localparam APP_MASK_WIDTH = 64;

    wire                      ui_clk, ui_clk_rst;
    wire                      c0_init_calib_complete;
    wire [APP_ADDR_WIDTH-1:0] c0_ddr4_app_addr = 0;
    wire [2:0]                c0_ddr4_app_cmd = 0;
    wire                      c0_ddr4_app_en = 0;
    wire [APP_DATA_WIDTH-1:0] c0_ddr4_app_wdf_data = 0;
    wire                      c0_ddr4_app_wdf_end = 0;
    wire [APP_MASK_WIDTH-1:0] c0_ddr4_app_wdf_mask = 0;
    wire                      c0_ddr4_app_wdf_wren = 0;
    wire [APP_DATA_WIDTH-1:0] c0_ddr4_app_rd_data;
    wire                      c0_ddr4_app_rd_data_end;
    wire                      c0_ddr4_app_rd_data_valid;
    wire                      c0_ddr4_app_rdy;
    wire                      c0_ddr4_app_wdf_rdy;

    localparam UART_BIT_WIDTH = 8;

    logic [UART_BIT_WIDTH-1:0] uart_tx_data;
    logic                      uart_tx_en, tx_rdy_n;
    logic [UART_BIT_WIDTH-1:0] uart_rx_data;
    logic                      uart_rx_valid;

    /* UART controller */
    uart #(
        .BITWIDTH(UART_BIT_WIDTH)
    ) uart_control (
        .clk      (ui_clk),
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

    always_ff @(posedge ui_clk) begin
        if (uart_rx_valid) begin
            uart_tx_data <= uart_rx_data;
            uart_tx_en   <= 1;
        end else begin
            uart_tx_en <= 0;
        end
    end

    /* Dram controller */
    ddr4_4g_x64 ddr4_control (
      .sys_rst                (sys_rst),

      .c0_sys_clk_p              (c0_sys_clk_p),
      .c0_sys_clk_n              (c0_sys_clk_n),
      .c0_init_calib_complete    (c0_init_calib_complete),

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

      .c0_ddr4_app_addr          (c0_ddr4_app_addr),
      .c0_ddr4_app_cmd           (c0_ddr4_app_cmd),
      .c0_ddr4_app_en            (c0_ddr4_app_en),
      .c0_ddr4_app_hi_pri        (1'b0),
      .c0_ddr4_app_wdf_data      (c0_ddr4_app_wdf_data),
      .c0_ddr4_app_wdf_end       (c0_ddr4_app_wdf_end),
      .c0_ddr4_app_wdf_mask      (c0_ddr4_app_wdf_mask),
      .c0_ddr4_app_wdf_wren      (c0_ddr4_app_wdf_wren),
      .c0_ddr4_app_rd_data       (c0_ddr4_app_rd_data),
      .c0_ddr4_app_rd_data_end   (c0_ddr4_app_rd_data_end),
      .c0_ddr4_app_rd_data_valid (c0_ddr4_app_rd_data_valid),
      .c0_ddr4_app_rdy           (c0_ddr4_app_rdy),
      .c0_ddr4_app_wdf_rdy       (c0_ddr4_app_wdf_rdy),
      .c0_ddr4_ui_clk            (ui_clk),
      .c0_ddr4_ui_clk_sync_rst   (ui_clk_rst),

      .dbg_clk                   (),
      .dbg_bus                   ()
    );
    
endmodule


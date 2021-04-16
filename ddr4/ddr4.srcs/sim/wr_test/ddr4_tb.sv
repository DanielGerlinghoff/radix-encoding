`timescale 1ps / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 13/04/2021
// 
// Description: Test of DDR4 DRAM transmissions
// 
//////////////////////////////////////////////////////////////////////////////////

module ddr4_tb;
    localparam APP_ADDR_WIDTH   = 29;
    localparam nCK_PER_CLK      = 4;
    localparam APP_DATA_WIDTH   = 512;
    localparam APP_MASK_WIDTH   = 64;
    localparam MEM_ADDR_ORDER   = "ROW_COLUMN_BANK";
    localparam ROW_WIDTH        = 15;
    localparam COL_WIDTH        = 10;
    localparam BANK_WIDTH       = 2;
    localparam BANK_GROUP_WIDTH = 2;
    localparam S_HEIGHT		    = 1;
    localparam MEMORY_WIDTH     = 8;

    /* Signals */
    logic [APP_ADDR_WIDTH-1:0] c0_ddr4_app_addr;
    logic [2:0]                c0_ddr4_app_cmd;
    logic                      c0_ddr4_app_en;
    logic [APP_DATA_WIDTH-1:0] c0_ddr4_app_wdf_data;
    logic                      c0_ddr4_app_wdf_end;
    logic [APP_MASK_WIDTH-1:0] c0_ddr4_app_wdf_mask;
    logic                      c0_ddr4_app_wdf_wren;
    logic [APP_DATA_WIDTH-1:0] c0_ddr4_app_rd_data;
    logic                      c0_ddr4_app_rd_data_end;
    logic                      c0_ddr4_app_rd_data_valid;
    logic                      c0_ddr4_app_rdy;
    logic                      c0_ddr4_app_wdf_rdy;
    logic                      ui_clk,ui_clk_rst;

    reg          sys_clk_i;
    reg          sys_rst;
    wire         c0_sys_clk_p;
    wire         c0_sys_clk_n;

    wire         c0_ddr4_act_n;
    wire  [16:0] c0_ddr4_adr;
    wire  [1:0]  c0_ddr4_ba;
    wire  [1:0]  c0_ddr4_bg;
    wire  [0:0]  c0_ddr4_cke;
    wire  [0:0]  c0_ddr4_odt;
    wire  [0:0]  c0_ddr4_cs_n;
    wire         c0_ddr4_ck_t;
    wire         c0_ddr4_ck_c;
    wire         c0_ddr4_reset_n;
    wire  [7:0]  c0_ddr4_dm_dbi_n;
    wire  [63:0] c0_ddr4_dq;
    wire  [7:0]  c0_ddr4_dqs_c;
    wire  [7:0]  c0_ddr4_dqs_t;
    wire         c0_init_calib_complete;

    bit          en_model;
    tri          model_enable = en_model;

    initial
        sys_clk_i = 1'b0;
    always
        sys_clk_i = #(4998/2.0) ~sys_clk_i;

    initial begin
        sys_rst = 1'b0;
        #200
        sys_rst = 1'b1;
        en_model = 1'b0; 
        #5 en_model = 1'b1;
        #200;
        sys_rst = 1'b0;
        #100;
    end

    assign c0_sys_clk_p = sys_clk_i;
    assign c0_sys_clk_n = ~sys_clk_i;

    initial begin
        c0_ddr4_app_addr     = 0;
        c0_ddr4_app_cmd      = 7;
        c0_ddr4_app_en       = 0;
        c0_ddr4_app_wdf_data = 0;
        c0_ddr4_app_wdf_end  = 0;
        c0_ddr4_app_wdf_mask = {~60'b0, 4'b0};
        c0_ddr4_app_wdf_wren = 0;

        wait (c0_ddr4_app_rdy);

        /* Write data */
        for (int i = 0; i < 100; i++) begin
            @(negedge ui_clk);
            if (!c0_ddr4_app_rdy) begin
                c0_ddr4_app_en       = 0;
                c0_ddr4_app_wdf_wren = 0;
                i--;

            end else begin
                c0_ddr4_app_en       = 1;
                c0_ddr4_app_cmd      = 0;
                c0_ddr4_app_addr     = i * 8;
                c0_ddr4_app_wdf_wren = 1;
                c0_ddr4_app_wdf_end  = 1;
                c0_ddr4_app_wdf_data = $random();
            end
        end

        @(negedge ui_clk);
        c0_ddr4_app_en       = 0;
        c0_ddr4_app_wdf_wren = 0;
        c0_ddr4_app_wdf_end  = 0;

        /* Read data */
        for (int i = 0; i < 100; i++) begin
            @(negedge ui_clk);
            c0_ddr4_app_en       = 1;
            c0_ddr4_app_cmd      = 1;
            c0_ddr4_app_addr     = i * 8;
        end

        @(negedge ui_clk);
        c0_ddr4_app_en        = 0;

        #200ns $finish;

    end

    /* DRAM controller */
    ddr4_4g_x64 u_ddr4_4g_x64 (
      .sys_rst                   (sys_rst),

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
      .c0_ddr4_ui_clk_sync_rst   (ui_clk_rst)
    );

    /* DRAM model */
    ddr4_wrapper mem(
        .model_enable     (model_enable),
        .c0_ddr4_act_n    (c0_ddr4_act_n),
        .c0_ddr4_adr      (c0_ddr4_adr),
        .c0_ddr4_ba       (c0_ddr4_ba),
        .c0_ddr4_bg       (c0_ddr4_bg),
        .c0_ddr4_cke      (c0_ddr4_cke),
        .c0_ddr4_odt      (c0_ddr4_odt),
        .c0_ddr4_cs_n     (c0_ddr4_cs_n),
        .c0_ddr4_ck_t     (c0_ddr4_ck_t),
        .c0_ddr4_ck_c     (c0_ddr4_ck_c),
        .c0_ddr4_reset_n  (c0_ddr4_reset_n),
        .c0_ddr4_dm_dbi_n (c0_ddr4_dm_dbi_n),
        .c0_ddr4_dq       (c0_ddr4_dq),
        .c0_ddr4_dqs_c    (c0_ddr4_dqs_c),
        .c0_ddr4_dqs_t    (c0_ddr4_dqs_t)
    );

endmodule


`timescale 1ns / 1ps
`default_nettype none
`include "sim.vh"
`define SKIP_UART
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 19/05/2021
//
// Description: Test of design module top
//
//////////////////////////////////////////////////////////////////////////////////


module top_tb;
    /* Clock signal */
    logic clk;
    logic clk_half;
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    initial begin
        clk_half = 0;
        forever #(CLK_PERIOD) clk_half = ~clk_half;
    end

    /* Module input signals */
    int          act_file;
    logic [0:31] input_data;
    logic        uart_tx_en;
    logic [7:0]  uart_tx_data;
    logic        uart_txd;

    initial begin
        uart_tx_en = 0;
        uart_tx_data = 0;

        #50us;
        uart_tx_en = 1;
        uart_tx_data = "R";
        #(2*CLK_PERIOD);
        uart_tx_en = 0;

`ifndef SKIP_UART #44100ns;
`else   #(2*CLK_PERIOD); `endif
        uart_tx_en = 1;
        uart_tx_data = "I";
        #(2*CLK_PERIOD);
        uart_tx_en = 0;

        act_file = $fopen("bram_activation.mif", "r");
        for (int row = 0; row < 3 * 32; row++) begin
            $fscanf(act_file, "%b", input_data);
            for (int val = 0; val < 32; val+=8) begin
`ifndef SKIP_UART #44100ns;
`else           #(2*CLK_PERIOD); `endif
                uart_tx_en = 1;
                uart_tx_data = input_data[val+:8];
                #(2*CLK_PERIOD);
                uart_tx_en = 0;
            end
        end

`ifndef SKIP_UART #44100ns;
`else   #(4*CLK_PERIOD); `endif
        uart_tx_en = 1;
        uart_tx_data = "S";
        #(2*CLK_PERIOD);
        uart_tx_en = 0;

    end

`ifndef SKIP_UART
    uart #(
        .CLKPERIOD (10e-9),
        .STOPBITS  (2)
    ) uart_ctrl (
        .clk      (clk_half),
        .rstn     (),
        .rxd      (),
        .cts      (),
        .txd      (uart_txd),
        .rts      (),

        .tx_data  (uart_tx_data),
        .tx_en    (uart_tx_en),
        .tx_rdy_n (),
        .rx_data  (),
        .rx_valid ()
    );
`endif

    /* Module instantiation */
    top test (
        .sys_clk_p (clk),
        .sys_clk_n (~clk),

`ifdef SKIP_UART
// NOTE: Add those ports and remove UART controller in module top
        .uart_rx_data  (uart_tx_data),
        .uart_rx_valid (uart_tx_en),
        .uart_tx_en    (),
        .uart_tx_data  (),
`endif

        .uart_rst_n (),
        .uart_txd   (uart_txd),
        .uart_rts   (0),
        .uart_rxd   (),
        .uart_cts   ()
    );

endmodule


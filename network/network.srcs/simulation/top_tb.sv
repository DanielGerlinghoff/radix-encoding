`timescale 1ns / 1ps
`default_nettype none
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
    localparam real CLK_PERIOD = 5.0;

    logic clk;
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    /* Module input signals */
    localparam UART_DELAY = int'(CLK_PERIOD * 2500);

    int           act_file;
    logic [0:31]  input_data;
    logic [0:511] dram_data;
    logic         uart_tx_en;
    logic [7:0]   uart_tx_data;
    logic         uart_txd;

    initial begin
        uart_tx_en = 0;
        uart_tx_data = 0;

        #50us;
        uart_tx_en = 1;
        uart_tx_data = "R";
        #(CLK_PERIOD);
        uart_tx_en = 0;
`ifndef SKIP_UART #120us;
`else   #(10000*CLK_PERIOD); `endif

`ifdef DRAM_ENABLE
`ifndef SKIP_UART #(UART_DELAY);
`else   #(CLK_PERIOD); `endif
        uart_tx_en = 1;
        uart_tx_data = "D";
        #(CLK_PERIOD);
        uart_tx_en = 0;

        act_file = $fopen("dram_kernel.mif", "r");
        for (int row = 0; row < 710; row++) begin
            $fscanf(act_file, "%b", dram_data);
            for (int val = 0; val < 512; val+=8) begin
`ifndef SKIP_UART #(UART_DELAY);
`else           #(CLK_PERIOD); `endif
                uart_tx_en = 1;
                uart_tx_data = dram_data[val+:8];
                #(CLK_PERIOD);
                uart_tx_en = 0;
            end
        end
`endif

`ifndef SKIP_UART #(UART_DELAY);
`else   #(CLK_PERIOD); `endif
        uart_tx_en = 1;
        uart_tx_data = "I";
        #(CLK_PERIOD);
        uart_tx_en = 0;

        act_file = $fopen("bram_activation.mif", "r");
        for (int row = 0; row < 3 * 32; row++) begin
            $fscanf(act_file, "%b", input_data);
            for (int val = 0; val < 32; val+=8) begin
`ifndef SKIP_UART #(UART_DELAY);
`else           #(CLK_PERIOD); `endif
                uart_tx_en = 1;
                uart_tx_data = input_data[val+:8];
                #(CLK_PERIOD);
                uart_tx_en = 0;
            end
        end

`ifndef SKIP_UART #(UART_DELAY);
`else   #(CLK_PERIOD); `endif
        uart_tx_en = 1;
        uart_tx_data = "S";
        #(CLK_PERIOD);
        uart_tx_en = 0;

    end

`ifndef SKIP_UART
    uart_ctrl uart_control (
        .clk      (clk),
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
//       Disable DRAM and remove respective ports
        .uart_rx_data  (uart_tx_data),
        .uart_rx_valid (uart_tx_en),
        .uart_tx_data  (),
        .uart_tx_en    (),
`endif

        .uart_txd   (uart_txd),
        .uart_rts   (0),
        .uart_rxd   (),
        .uart_cts   ()
    );

endmodule

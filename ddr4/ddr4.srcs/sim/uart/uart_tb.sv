`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 16/04/2021
// 
// Description: Test of UART controller
// 
//////////////////////////////////////////////////////////////////////////////////


module uart_tb;
    /* Module parameters */
    localparam BITWIDTH   = 8;
    localparam BAUDRATE   = 921600;
    localparam BAUDPERIOD = 1e9 / BAUDRATE;
    localparam CLKPERIOD  = 4.284;

    /* Module input signals */
    logic clk;
    logic rxd, cts;
    logic txd, rts;
    logic rstn;

    logic [BITWIDTH-1:0] tx_data;
    logic       tx_en;
    logic       tx_rdy_n;
    logic [BITWIDTH-1:0] rx_data;
    logic       rx_valid;

    initial begin
        clk = 0;
        forever begin
            #(CLKPERIOD/2) clk = ~clk;
        end
    end

    initial begin
        rxd = 1;
        cts = 0;

        #(BAUDPERIOD);
        rxd = 0;
        #(BAUDPERIOD);
        rxd = 1;
        #(BAUDPERIOD);
        rxd = 0;
        #(BAUDPERIOD);
        rxd = 0;
        #(BAUDPERIOD);
        rxd = 0;
        #(BAUDPERIOD);
        rxd = 0;
        #(BAUDPERIOD);
        rxd = 1;
        #(BAUDPERIOD);
        rxd = 1;
        #(BAUDPERIOD);
        rxd = 0;
        #(BAUDPERIOD);
        rxd = 1;
    end

    initial begin
        tx_data = 0;
        tx_en   = 0;

        #(10*CLKPERIOD);
        tx_data = 8'h61;
        tx_en   = 1;

        #(CLKPERIOD);
        tx_data = 8'h62;
        tx_en   = 1;

        #(CLKPERIOD);
        tx_data = 8'h63;
        tx_en   = 1;

        #(CLKPERIOD);
        tx_data = 8'h64;
        tx_en   = 1;

        #(CLKPERIOD);
        tx_data = 0;
        tx_en   = 0;

        #((4*10+1)*BAUDPERIOD);
        $finish;
    end

    /* Module instantiation */
    uart #(
        .BITWIDTH(BITWIDTH),
        .BAUDRATE(BAUDRATE),
        .CLKPERIOD(CLKPERIOD/1e9)
    ) test (
        .*
    );

endmodule


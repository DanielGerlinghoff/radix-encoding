`timescale 1ns / 1ps
`default_nettype none
`include "../sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 29/03/2021
// 
// Description: Test of design module conv_accumulator
// 
//////////////////////////////////////////////////////////////////////////////////


module conv_accumulator_tb;
    /* Clock signal */
    logic clk;
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end

    /* Module parameters */
    localparam INP_SIZE = 8;
    localparam OUT_SIZE = 12;

    /* Module input signals */
    logic [INP_SIZE-1:0] addend;
    logic [OUT_SIZE-1:0] accumulator;
    logic select, clear;

    initial begin
        addend = 0;
        select = 0;

        #(RST_PERIOD) clear = 1;
        #(CLK_PERIOD) clear = 0;

        #(50);
        for (addend = 0; addend < 10; addend++) begin
            #(CLK_PERIOD);
            select = ~select;
        end

        #(50) clear = 1;
        #(CLK_PERIOD) clear = 0;
    end

    /* Module instantiation */
    conv_accumulator #(
        .INP_SIZE(INP_SIZE),
        .OUT_SIZE(OUT_SIZE)
    ) test (
        .*
    );

endmodule

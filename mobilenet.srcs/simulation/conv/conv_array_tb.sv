`timescale 1ns / 1ps
`default_nettype none
`include "../sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 31/03/2021
// 
// Description: Test of design module conv_array
// 
//////////////////////////////////////////////////////////////////////////////////


module conv_array_tb;
    /* Clock signal */
    logic clk;
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end
    
    /* Module parameters */
    localparam ROWS     = 3;
    localparam COLS     = 4;
    localparam KER_SIZE = 8;
    localparam KER_VALS = 9;
    localparam SUM_SIZE = 12;

    /* Module input signals */
    logic start, stop;
    logic activation [ROWS][COLS];
    logic [SUM_SIZE-1:0] row_conv [COLS];

    if_kernel #(
        .KER_SIZE(KER_SIZE),
        .KER_VALS(KER_VALS)
    ) ker ();

    initial begin
        start      = 0;
        stop       = 0;
        activation = '{default: 0};
        ker.data   = 0;
        ker.wren   = 0;

        #(RST_PERIOD);
        for (int k = 0; k < 1; k++) begin
            for (int val = 0; val < KER_VALS; val++) begin
                ker.data[val] = $random();
            end
            ker.wren = 1'b1;
            #(CLK_PERIOD);
            ker.wren = 1'b0;
            #(CLK_PERIOD);
        end

        #(RST_PERIOD) start = 1;
        activation = '{default: 1};
        #(CLK_PERIOD) start = 0;

        #(40*6-CLK_PERIOD) stop = 1;
        #(CLK_PERIOD)      stop = 0;

    end

    /* Module instantiation */
    conv_array #(
        .ROWS(ROWS),
        .COLS(COLS),
        .KER_SIZE(KER_SIZE),
        .KER_VALS(KER_VALS),
        .SUM_SIZE(SUM_SIZE)
    ) test (
        .*
    );

endmodule


`timescale 1ns / 1ps
`default_nettype none
`include "../sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 29/03/2021
// 
// Description: Test of design module conv_row
// 
//////////////////////////////////////////////////////////////////////////////////


module conv_row_tb;
    /* Clock signal */
    logic clk;
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end

    /* Module parameters */
    localparam COLS     = 4;
    localparam KER_SIZE = 8;
    localparam SUM_SIZE = 8;

    /* Module input signals */
    logic clear, enable, activation [COLS];
    logic [KER_SIZE-1:0] kernel [COLS];
    logic [SUM_SIZE-1:0] sum_in [COLS], sum_out [COLS];
    logic sum_wren;

    initial begin
        activation = '{default: 0};
        kernel     = '{default: 8};
        enable     = 0;
        clear      = 0;
        sum_wren   = 0;

        #(RST_PERIOD) clear = 1;
        #(CLK_PERIOD) clear = 0;

        #(RST_PERIOD);
        enable = 1;
        for (int i = 0; i < 10; i++) begin
            #(CLK_PERIOD);
            for (int col = 0; col < COLS; col++) begin
                activation[col] = $urandom_range(1);
            end
        end
        enable = 0;

        #(RST_PERIOD) sum_wren = 1;
        for (int col = 0; col < COLS; col++) begin
            sum_in[col] = $random();
        end
        #(CLK_PERIOD) sum_wren = 0;

        #(RST_PERIOD) clear = 1;
        #(CLK_PERIOD) clear = 0;
    end

    /* Module instantiation */
    conv_row #(
        .COLS(COLS),
        .KER_SIZE(KER_SIZE),
        .SUM_SIZE(SUM_SIZE)
    ) test (
        .*
    );

endmodule


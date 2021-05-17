`timescale 1ns / 1ps
`default_nettype none
`include "sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 04/05/2021
//
// Description: Test of design module processor
//
//////////////////////////////////////////////////////////////////////////////////


module processor_tb;
    /* Clock signal */
    logic clk;
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end

    /* Memory initialization */
    initial begin
        test.bram.rom[ 0] = {4'h1, 5'h00, 23'h1};        // CONF UNIT EN
        test.bram.rom[ 1] = {4'h1, 5'h1f, 4'h0, 19'h1};  // CONF ALL PAR
        test.bram.rom[ 2] = {4'h1, 5'h1f, 4'h1, 19'h0};  // CONF ALL STR
        test.bram.rom[ 3] = {4'h1, 5'h1f, 4'h2, 19'h0};  // CONF ALL PAD
        test.bram.rom[ 4] = {4'h2, 28'h0};               // RST
        test.bram.rom[ 5] = {4'h1, 5'h1f, 4'h4, 19'h0};  // CONF ALL KSEL
        test.bram.rom[ 6] = {4'h4, 5'h00, 23'h0};        // KERL UNIT ADDR
        test.bram.rom[ 7] = {4'h4, 5'h00, 23'h1};        // KERL UNIT ADDR
        test.bram.rom[ 8] = {4'h1, 5'h1f, 4'h5, 19'h0};  // CONF ALL ASEL
        test.bram.rom[ 9] = {4'h1, 5'h1f, 4'h3, 19'h3};  // CONF ALL OUT DEL
        test.bram.rom[10] = {4'h5, 5'h00, 23'h0};        // ACTL UNIT ADDR
        test.bram.rom[11] = {4'h3, 28'h0};               // PROC
        test.bram.rom[12] = {4'h5, 5'h00, 23'h1};        // ACTL UNIT ADDR
        test.bram.rom[13] = {4'h7, 5'h00, 2'h0, 21'h0};  // WAIT UNIT CONV
    end

    initial begin
        mem_ker.gen_conv_bram[0].bram_i.ram[0] = {9 {8'h01}};
        mem_ker.gen_conv_bram[0].bram_i.ram[1] = {9 {8'h02}};

        mem_act.gen_bram[0].bram_i.ram[0] = {38 {3'b010}};
        mem_act.gen_bram[0].bram_i.ram[1] = {38 {3'b101}};
        mem_act.gen_bram[0].bram_i.ram[2] = {38 {3'b010}};
        mem_act.gen_bram[0].bram_i.ram[3] = {38 {3'b101}};
    end

    /* Module input signals */
    logic reset;
    logic start, finish;

    if_configuration conf ();
    if_control ctrl ();
    if_activation act (.clk);
    if_kernel ker (.clk);

    initial begin
        reset = 0;
        start = 0;

        #(CLK_PERIOD) reset = 1;
        #(CLK_PERIOD) reset = 0;

        #(RST_PERIOD) start = 1;
        #(CLK_PERIOD) start = 0;

        wait(test.instr_op[1] == test.WAIT);
        #(4*CLK_PERIOD);
        ctrl.finish[0] = 1;

        #(2*CLK_PERIOD);
        $finish();
    end

    /* Module instantiation */
    processor test (
        .*
    );

    mem_kernel mem_ker (.*);
    mem_activation mem_act (.*);

endmodule


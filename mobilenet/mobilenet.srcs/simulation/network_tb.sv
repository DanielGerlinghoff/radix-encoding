`timescale 1ns / 1ps
`default_nettype none
`include "sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 29/04/2021
//
// Description: Test of design module network
//
//////////////////////////////////////////////////////////////////////////////////


module network_tb;
    /* Clock signal */
    logic clk;
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end

    /* Module parameters */
    localparam ID_CONV = 0;
    localparam ID_KER  = 0;
    localparam ID_ACT  = 0;
    localparam KER     = pkg_processing::KER_SIZE[ID_CONV];

    /* Memory initialization */
    initial begin
        test.proc.bram.rom[ 0] = {4'h1, 5'h00, 23'h1};        // CONF UNIT EN
        test.proc.bram.rom[ 1] = {4'h1, 5'h1f, 3'h0, 20'h1};  // CONF ALL PAR
        test.proc.bram.rom[ 2] = {4'h1, 5'h1f, 3'h1, 20'h0};  // CONF ALL STR
        test.proc.bram.rom[ 3] = {4'h2, 28'h0};               // RST
        test.proc.bram.rom[ 4] = {4'h4, 5'h00, 23'h0};        // KERL UNIT ADDR
        test.proc.bram.rom[ 5] = {4'h4, 5'h00, 23'h1};        // KERL UNIT ADDR
        test.proc.bram.rom[ 6] = {4'h1, 5'h1f, 3'h3, 20'h0};  // CONF ALL MEM
        test.proc.bram.rom[ 8] = {4'h1, 5'h1f, 3'h2, 20'h3};  // CONF ALL OUT DEL
        test.proc.bram.rom[ 7] = {4'h5, 5'h00, 23'h0};        // ACTL UNIT ADDR
        test.proc.bram.rom[ 9] = {4'h3, 28'h0};               // PROC
        test.proc.bram.rom[10] = {4'h5, 5'h00, 23'h1};        // ACTL UNIT ADDR
        test.proc.bram.rom[11] = {4'h7, 5'h00, 2'h0, 21'h0};  // WAIT UNIT CONV
        test.proc.bram.rom[12] = {4'h3, 28'h0};               // PROC
        test.proc.bram.rom[13] = {4'h5, 5'h00, 23'h2};        // ACTL UNIT ADDR
        test.proc.bram.rom[14] = {4'h7, 5'h00, 2'h0, 21'h0};  // WAIT UNIT CONV
        test.proc.bram.rom[15] = {4'h3, 28'h0};               // PROC
        test.proc.bram.rom[16] = {4'h5, 5'h00, 23'h3};        // ACTL UNIT ADDR
        test.proc.bram.rom[17] = {4'h1, 5'h1f, 3'h2, 20'h2};  // CONF ALL OUT DIR
        test.proc.bram.rom[18] = {4'h7, 5'h00, 2'h0, 21'h0};  // WAIT UNIT CONV
        test.proc.bram.rom[19] = {4'h3, 28'h0};               // PROC
        test.proc.bram.rom[20] = {4'h7, 5'h00, 2'h0, 21'h0};  // WAIT UNIT CONV
        test.proc.bram.rom[21] = {4'h2, 28'h0};               // RST
        test.proc.bram.rom[22] = {4'h1, 5'h1f, 3'h2, 20'h3};  // CONF ALL OUT DEL
        test.proc.bram.rom[23] = {4'h5, 5'h00, 23'h0};        // ACTL UNIT ADDR
        test.proc.bram.rom[24] = {4'h3, 28'h0};               // PROC
        test.proc.bram.rom[25] = {4'h5, 5'h00, 23'h1};        // ACTL UNIT ADDR
        test.proc.bram.rom[26] = {4'h7, 5'h00, 2'h0, 21'h0};  // WAIT UNIT CONV
        test.proc.bram.rom[27] = {4'h3, 28'h0};               // PROC
        test.proc.bram.rom[28] = {4'h5, 5'h00, 23'h2};        // ACTL UNIT ADDR
        test.proc.bram.rom[29] = {4'h7, 5'h00, 2'h0, 21'h0};  // WAIT UNIT CONV
        test.proc.bram.rom[30] = {4'h3, 28'h0};               // PROC
        test.proc.bram.rom[31] = {4'h5, 5'h00, 23'h3};        // ACTL UNIT ADDR
        test.proc.bram.rom[32] = {4'h1, 5'h1f, 3'h2, 20'h1};  // CONF ALL OUT ADD
        test.proc.bram.rom[33] = {4'h7, 5'h00, 2'h0, 21'h0};  // WAIT UNIT CONV
        test.proc.bram.rom[34] = {4'h3, 28'h0};               // PROC
        test.proc.bram.rom[35] = {4'h7, 5'h00, 2'h0, 21'h0};  // WAIT UNIT CONV
        test.proc.bram.rom[36] = {4'h2, 28'h0};               // RST
        test.proc.bram.rom[37] = {4'h1, 5'h1f, 3'h2, 20'h3};  // CONF ALL OUT DEL
        test.proc.bram.rom[38] = {4'h5, 5'h00, 23'h0};        // ACTL UNIT ADDR
        test.proc.bram.rom[39] = {4'h3, 28'h0};               // PROC
        test.proc.bram.rom[40] = {4'h5, 5'h00, 23'h1};        // ACTL UNIT ADDR
        test.proc.bram.rom[41] = {4'h7, 5'h00, 2'h0, 21'h0};  // WAIT UNIT CONV
        test.proc.bram.rom[42] = {4'h3, 28'h0};               // PROC
        test.proc.bram.rom[43] = {4'h5, 5'h00, 23'h2};        // ACTL UNIT ADDR
        test.proc.bram.rom[44] = {4'h7, 5'h00, 2'h0, 21'h0};  // WAIT UNIT CONV
        test.proc.bram.rom[45] = {4'h3, 28'h0};               // PROC
        test.proc.bram.rom[46] = {4'h5, 5'h00, 23'h3};        // ACTL UNIT ADDR
        test.proc.bram.rom[47] = {4'h1, 5'h1f, 3'h2, 20'h0};  // CONF ALL OUT SFT
        test.proc.bram.rom[48] = {4'h7, 5'h00, 2'h0, 21'h0};  // WAIT UNIT CONV
        test.proc.bram.rom[49] = {4'h3, 28'h0};               // PROC
        test.proc.bram.rom[50] = {4'h7, 5'h00, 2'h1, 21'h0};  // WAIT UNIT CWR
        test.proc.bram.rom[51] = {4'h1, 5'h1f, 3'h3, 20'h1};  // CONF ALL MEM
        test.proc.bram.rom[52] = {4'h1, 5'h1f, 3'h4, 20'h4};  // CONF ALL SCL
        test.proc.bram.rom[53] = {4'h1, 5'h1f, 3'h5, 20'h4};  // CONF ALL ASTF
        test.proc.bram.rom[54] = {4'h1, 5'h1f, 3'h6, 20'ha};  // CONF ALL ASTB
        test.proc.bram.rom[55] = {4'h6, 5'h00, 23'h0};        // ACTS UNIT ADDR
        test.proc.bram.rom[56] = {4'h7, 5'h00, 2'h2, 21'h0};  // WAIT UNIT TRAN
        test.proc.bram.rom[57] = {4'h8, 28'h0};               // END
    end

    initial begin
        test.mem_ker.gen_bram[ID_KER].bram_i.ram[0] = {9 {8'h01}};
        test.mem_ker.gen_bram[ID_KER].bram_i.ram[1] = {9 {8'h02}};

        test.mem_act.gen_bram[ID_ACT].bram_i.ram[0] = {38 {3'b010}};
        test.mem_act.gen_bram[ID_ACT].bram_i.ram[1] = {38 {3'b101}};
        test.mem_act.gen_bram[ID_ACT].bram_i.ram[2] = {38 {3'b010}};
        test.mem_act.gen_bram[ID_ACT].bram_i.ram[3] = {38 {3'b101}};
    end

    /* Module input signals */
    logic proc_reset, proc_start, proc_finish;
    logic dram_enable;
    logic [pkg_memory::DRAM_ADDR_BITS-1:0] dram_addr;
    logic [pkg_memory::DRAM_DATA_BITS-1:0] dram_data;

    initial begin
        proc_reset = 0;
        proc_start = 0;

        #(CLK_PERIOD) proc_reset = 1;
        #(CLK_PERIOD) proc_reset = 0;

        #(RST_PERIOD) proc_start = 1;
        #(CLK_PERIOD) proc_start = 0;

        wait(proc_finish);
        #(4*CLK_PERIOD);
        $finish();

    end

    /* Module instantiation */
    network test (
        .*
    );

endmodule


`timescale 1ns / 1ps
`default_nettype none
`include "../sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 29/04/2021
//
// Description: Test of design module mem_kernel
//
//////////////////////////////////////////////////////////////////////////////////


module mem_kernel_tb;
    import pkg_memory::*;

    /* Clock signal */
    logic clk;
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end

    /* Module parameters */
    localparam ID = 0;

    /* Module input signals */
    if_kernel ker (
        .clk
    );

    initial begin
        ker.bram_wr_en[ID] = 0;
        ker.bram_rd_en[ID] = 0;

        #(RST_PERIOD);
        for (int h = 0; h < KER_HEIGHT_MAX[1]; h++) begin
            for (int w = 0; w < DRAM_DATA_BITS; w++)
                ker.bram_wr_data[w] = $random();
            ker.bram_wr_addr = h;
            ker.bram_wr_en[ID] = 1;
            #(CLK_PERIOD);
        end

        ker.bram_wr_en[ID] = 0;

        #(RST_PERIOD);
        for (int h = 0; h < KER_HEIGHT[ID]; h++) begin
            ker.bram_rd_addr = h;
            ker.bram_rd_en[ID] = 1;
            #(CLK_PERIOD);
        end

        ker.bram_rd_en[ID] = 0;

        #(RST_PERIOD);
        $finish();

    end

    /* Module instantiation */
    mem_kernel test (
        .*
    );

endmodule


`timescale 1ns / 1ps
`default_nettype none
`include "../sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 29/04/2021
//
// Description: Test of design module mem_activation
//
//////////////////////////////////////////////////////////////////////////////////


module mem_activation_tb;
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
    if_activation act (
        .clk (clk)
    );

    initial begin
        act.wr_en_u[ID] = '{default: 0};
        act.rd_en     = '{default: 0};

        for (int mem = 0; mem < ACT_NUM; mem++) begin
            #(RST_PERIOD);
            for (int val = 0; val < 4; val++) begin
                act.wr_addr_u[ID]    = val;
                act.wr_data_u[ID]    = $random();
                act.wr_en_u[ID][mem] = 1;
                #(CLK_PERIOD);
            end
            act.wr_en[ID][mem] = 0;

            #(RST_PERIOD);
            for (int val = 0; val < 4; val++) begin
                act.rd_addr    = val;
                act.rd_en[mem] = 1;
                #(CLK_PERIOD);
            end
            act.rd_en[mem] = 0;
        end

        #(RST_PERIOD);
        $finish();
    end

    /* Module instantiation */
    mem_activation test (
        .*
    );

endmodule


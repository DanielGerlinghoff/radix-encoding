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

    /* Module input signals */
    if_activation act (
        .clk (clk)
    );

    logic                        act_wr_en [$size(act.wr_en)];
    logic [$high(act.wr_data):0] act_wr_data;

    initial begin
        act_wr_en = '{0, 0};
        act.rd_en = '{0, 0};

        for (int mem = 0; mem < ACT_NUM; mem++) begin
            #(RST_PERIOD);
            for (int val = 0; val < 4; val++) begin
                act.wr_addr    = val;
                act_wr_data    = $random();
                act_wr_en[mem] = 1;
                #(CLK_PERIOD);
            end
            act_wr_en[mem] = 0;

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

    assign act.wr_en   = act_wr_en;
    assign act.wr_data = act_wr_data;

    /* Module instantiation */
    mem_activation test (
        .*
    );

endmodule


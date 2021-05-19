`timescale 1ns / 1ps
`default_nettype none
`include "../sim.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 16/05/2021
//
// Description: Test of design module lin_unit
//
//////////////////////////////////////////////////////////////////////////////////


module lin_unit_tb;
    /* Clock signal */
    logic clk;
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end

    /* Module parameters */
    import pkg_linear::*;

    localparam ID     = 3;
    localparam ID_WGT = 0;
    localparam ID_ACT = 2;
    localparam CHN    = 5;

    /* Module input signals */
    if_control ctrl ();
    if_configuration conf ();
    if_kernel ker (.clk);
    if_activation act (.clk);

    initial begin
        conf.enable[ID]   = 1;
        conf.lin_channels = CHN;
        conf.act_scale    = 1;
        ker.wgt_select    = ID_WGT;
        act.mem_rd_select = ID_ACT;
        act.mem_wr_select = ID_ACT + 1;
        act.addr_step     = '{84, 167};

        #(RST_PERIOD) ctrl.reset = 1;
        #(CLK_PERIOD) ctrl.reset = 0;

        #(RST_PERIOD);
        for (int t = 0; t < ACT_BITS; t++) begin
            for (int c = 0; c < CHN; c++) begin
                for (int b = 0; b < pkg_memory::DRAM_DATA_BITS; b++) ker.wgt_bram_rd_data[ID_WGT][b] = $random();
                ker.wgt_bram_rd_val[ID_WGT] = 1;
                for (int s = 0; s < LIN_SIZE; s++) act.rd_data[ID_ACT][s] = $random();
                act.rd_val[ID_ACT] = 1;
                #(CLK_PERIOD);
            end
            act.rd_val[ID_ACT] = 0;
            ker.wgt_bram_rd_val[ID_WGT] = 0;
            #(CLK_PERIOD);
        end

        wait(|act.transfer_finish);
        #(2*CLK_PERIOD) $finish();
    end

    /* Module instantiation */
    lin_unit #(
        .ID(ID)
    ) test (
        .*
    );

endmodule


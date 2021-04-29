`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 29/04/2021
//
// Description: Automatically generated package with memory configurations
//
//////////////////////////////////////////////////////////////////////////////////


package pkg_memory;
    /* Kernel memory */
    localparam int KER_NUM = 1;
    localparam int KER_WIDTH [KER_NUM] = '{128};
    localparam int KER_WIDTH_MAX = 128;
    localparam int KER_HEIGHT [KER_NUM] = '{16};
    localparam int KER_HEIGHT_MAX [2] = '{16, 4};
    localparam [800:1] KER_INIT [KER_NUM] = '{""};
    localparam int DRAM_WIDTH = 512;

endpackage


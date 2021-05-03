`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 29/04/2021
//
// Description: Automatically generated package with configurations for kernel
//              and activation memories
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

    /* Activation memory */
    localparam int ACT_NUM = 2;
    localparam int ACT_WIDTH [ACT_NUM] = '{224, 224};
    localparam int ACT_WIDTH_MAX = 224;
    localparam int ACT_HEIGHT [ACT_NUM] = '{16, 16};
    localparam int ACT_HEIGHT_MAX = 16;

    /* External DRAM */
    localparam int DRAM_DATA_BITS = 512;
    localparam int DRAM_ADDR_BITS = 29;

endpackage


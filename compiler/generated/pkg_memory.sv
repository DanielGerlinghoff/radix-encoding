`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 14/05/2021
//
// Description: Automatically generated package with configurations for kernel
//              and activation memories
//
//////////////////////////////////////////////////////////////////////////////////


package pkg_memory;
	/* Kernel memory */
	localparam int KER_NUM = 3;
	localparam int KER_WIDTH [KER_NUM] = '{75, 75, 75};
	localparam int KER_WIDTH_MAX = 75;
	localparam int KER_HEIGHT [KER_NUM] = '{6, 96, 1920};
	localparam int KER_HEIGHT_MAX [2] = '{1920, 0};
	localparam [800:1] KER_INIT [KER_NUM] = '{"bram_kernel_00.mif", "bram_kernel_01.mif", "bram_kernel_02.mif"};

	/* Activation memory */
	localparam int ACT_NUM = 2;
	localparam int ACT_WIDTH [ACT_NUM] = '{32, 28};
	localparam int ACT_WIDTH_MAX = 32;
	localparam int ACT_HEIGHT [ACT_NUM] = '{252, 504};
	localparam int ACT_HEIGHT_MAX = 504;
	localparam string ACT_INIT = "bram_activation.mif";

	/* Instruction memory */
	localparam int INS_WIDTH = 32;
	localparam int INS_HEIGHT = 26243;
	localparam string INS_INIT = "bram_instruction.mif";

	/* External DRAM */
	localparam int DRAM_DATA_BITS = 512;
	localparam int DRAM_ADDR_BITS = 29;

endpackage

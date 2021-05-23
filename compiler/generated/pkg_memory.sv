`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 23/05/2021
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
	localparam int KER_HEIGHT [KER_NUM] = '{1920};
	localparam int KER_HEIGHT_MAX [2] = '{1920, 480};
	localparam [800:1] KER_INIT [KER_NUM] = '{""};

	/* Weight memory */
	localparam int WGT_NUM = 1;
	localparam int WGT_HEIGHT [WGT_NUM] = '{120};
	localparam int WGT_HEIGHT_MAX = 120;
	localparam [800:1] WGT_INIT [WGT_NUM] = '{""};

	/* Activation memory */
	localparam int ACT_NUM = 5;
	localparam int ACT_WIDTH [ACT_NUM] = '{32, 28, 1, 1, 10};
	localparam int ACT_WIDTH_MAX = 32;
	localparam int ACT_HEIGHT [ACT_NUM] = '{252, 504, 360, 252, 10};
	localparam int ACT_HEIGHT_MAX = 504;
	localparam [800:1] ACT_INIT = "bram_activation.mif";

	/* Instruction memory */
	localparam int INS_WIDTH = 32;
	localparam int INS_HEIGHT = 26306;
	localparam [800:1] INS_INIT = "bram_instruction.mif";

	/* External DRAM */
	localparam int DRAM_DATA_BITS = 512;
	localparam int DRAM_ADDR_BITS = 29;

endpackage

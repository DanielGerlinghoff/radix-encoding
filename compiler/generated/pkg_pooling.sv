`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 10/05/2021
//
// Description: Automatically generated package with config of processing units
//
//////////////////////////////////////////////////////////////////////////////////


package pkg_pooling;
	localparam int POOLUNITS = 1;
	localparam int POOL_SIZE [POOLUNITS] = '{32};
	localparam int POOL_SIZE_MAX = 32;
	localparam int ACT_BITS = 3;
	localparam int KER_SIZE [POOLUNITS] = '{2};

	localparam int PARALLEL_DIM [POOLUNITS][2] = '{'{3, 6}};
	localparam int PARALLEL_NUM [POOLUNITS][3] = '{'{1, 2, 6}};
	localparam int PARALLEL_WIDTH [POOLUNITS][3] = '{'{28, 10, 1}};
	localparam int PARALLEL_MAX [POOLUNITS] = '{6};
	localparam int PARALLEL_IN [POOLUNITS][3][6][2] = '{
		'{
			'{'{0, 31}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
			'{'{0, 13}, '{14, 27}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
			'{'{0, 4}, '{6, 10}, '{12, 16}, '{18, 22}, '{24, 28}, '{30, 34}}
		}
	};
	localparam int PARALLEL_OUT [POOLUNITS][3][6][2] = '{
		'{
			'{'{0, 27}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
			'{'{0, 9}, '{14, 23}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
			'{'{0, 0}, '{6, 6}, '{12, 12}, '{18, 18}, '{24, 24}, '{30, 30}}
		}
	};

endpackage

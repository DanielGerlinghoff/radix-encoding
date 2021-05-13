`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 13/05/2021
//
// Description: Automatically generated package with config of convolution units
//
//////////////////////////////////////////////////////////////////////////////////


package pkg_convolution;
	localparam int CONVUNITS = 2;
	localparam int CONV_SIZE [CONVUNITS] = '{31, 31};
	localparam int CONV_SIZE_MAX = 31;
	localparam int CONV_BITS = 8;
	localparam int ACT_BITS = 3;
	localparam int KER_BITS = 3;
	localparam int KER_SIZE [CONVUNITS] = '{5, 5};

	localparam int PARALLEL_DIM [CONVUNITS][2] = '{'{3, 6}, '{3, 6}};
	localparam int PARALLEL_NUM [CONVUNITS][3] = '{'{1, 2, 6}, '{1, 2, 6}};
	localparam int PARALLEL_MAX [CONVUNITS] = '{6, 6};
	localparam int PARALLEL_IN [CONVUNITS][3][6][2] = '{
		'{
			'{'{0, 31}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
			'{'{0, 13}, '{14, 27}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
			'{'{0, 4}, '{6, 10}, '{12, 16}, '{18, 22}, '{24, 28}, '{30, 34}}
		},
		'{
			'{'{0, 31}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
			'{'{0, 13}, '{14, 27}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
			'{'{0, 4}, '{6, 10}, '{12, 16}, '{18, 22}, '{24, 28}, '{30, 34}}
		}
	};
	localparam int PARALLEL_OUT [CONVUNITS][3][6][2] = '{
		'{
			'{'{0, 27}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
			'{'{0, 9}, '{14, 23}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
			'{'{0, 0}, '{6, 6}, '{12, 12}, '{18, 18}, '{24, 24}, '{30, 30}}
		},
		'{
			'{'{0, 27}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
			'{'{0, 9}, '{14, 23}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
			'{'{0, 0}, '{6, 6}, '{12, 12}, '{18, 18}, '{24, 24}, '{30, 30}}
		}
	};

	localparam int STRIDE_DIM [CONVUNITS] = '{1, 1};
	localparam int STRIDE [CONVUNITS][1] = '{
		'{1},
		'{1}
	};
	localparam int STRIDE_MAX [CONVUNITS] = '{1, 1};

endpackage

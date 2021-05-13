`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 13/05/2021
//
// Description: Automatically generated package with config of pooling units
//
//////////////////////////////////////////////////////////////////////////////////


package pkg_pooling;
	localparam int POOLUNITS = 2;
	localparam bit MAX_N_AVG = 1;
	localparam int POOL_SIZE [2:POOLUNITS+1] = '{14, 14};
	localparam int ACT_BITS = 3;
	localparam int KER_SIZE [2:POOLUNITS+1] = '{2, 2};

	localparam int PARALLEL_DIM [2:POOLUNITS+1][2] = '{'{2, 2}, '{2, 2}};
	localparam int PARALLEL_NUM [2:POOLUNITS+1][2] = '{'{1, 2}, '{1, 2}};
	localparam int PARALLEL_MAX [2:POOLUNITS+1] = '{2, 2};
	localparam int PARALLEL_IN [2:POOLUNITS+1][2][2][2] = '{
		'{
			'{'{0, 27}, '{0, 0}},
			'{'{0, 9}, '{10, 19}}
		},
		'{
			'{'{0, 27}, '{0, 0}},
			'{'{0, 9}, '{10, 19}}
		}
	};
	localparam int PARALLEL_OUT [2:POOLUNITS+1][2][2][2] = '{
		'{
			'{'{0, 13}, '{0, 0}},
			'{'{0, 4}, '{5, 9}}
		},
		'{
			'{'{0, 13}, '{0, 0}},
			'{'{0, 4}, '{5, 9}}
		}
	};

endpackage

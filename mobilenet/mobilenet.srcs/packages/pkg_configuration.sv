`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 22/04/2021
//
// Description: Automatically generated package with config of convolution units
//
//////////////////////////////////////////////////////////////////////////////////


package pkg_configuration;
    /* Convolution settings */
    localparam int CONVUNITS = 2;
    localparam int CONV_SIZE [CONVUNITS] = '{112, 112};
    localparam int CONV_BITS = 12;
    localparam int ACT_BITS = 4;
    localparam int ACT_SIZE [CONVUNITS] = '{112, 112};
    localparam int KERNEL_BITS = 8;
    localparam int KERNEL_SIZE [CONVUNITS] = '{3, 3};

    /* Convolution activation input */
    localparam int PARALLEL_DIM [2] = '{5, 16};
    localparam int PARALLEL_NUM [5] = '{1, 2, 4, 8, 16};
    localparam int PARALLEL [5][16][2] = '{
        '{'{1, 224}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
        '{'{1, 112}, '{115, 226}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
        '{'{1, 56}, '{59, 114}, '{117, 172}, '{175, 230}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
        '{'{1, 28}, '{31, 58}, '{61, 88}, '{91, 118}, '{121, 148}, '{151, 178}, '{181, 208}, '{211, 238}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
        '{'{1, 14}, '{17, 30}, '{33, 46}, '{49, 62}, '{65, 78}, '{81, 94}, '{97, 110}, '{113, 126}, '{129, 142}, '{145, 158}, '{161, 174}, '{177, 190}, '{193, 206}, '{209, 222}, '{225, 238}, '{241, 254}}};

    localparam int STRIDE_DIM = 2;
    localparam int STRIDE [2] = '{1, 2};
    localparam int STRIDE_MAX = 2;

endpackage


`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 22/04/2021
//
// Description: Automatically generated package with config of convolution units
//
//////////////////////////////////////////////////////////////////////////////////


package pkg_processing;
    /* Convolution settings */
    localparam int CONVUNITS = 1;
    localparam int CONV_SIZE [CONVUNITS] = '{128};
    localparam int CONV_SIZE_MAX = 128;
    localparam int CONV_BITS = 12;
    localparam int ACT_BITS = 4;
    localparam int KER_BITS = 8;
    localparam int KER_SIZE [CONVUNITS] = '{3};

    localparam int PARALLEL_DIM [CONVUNITS][2] = '{'{5, 16}};
    localparam int PARALLEL_NUM [CONVUNITS][5] = '{'{1, 2, 4, 8, 16}};
    localparam int PARALLEL_WIDTH [CONVUNITS][5] = '{'{112, 56, 28, 14, 7}};
    localparam int PARALLEL_MAX [CONVUNITS] = '{16};
    localparam int PARALLEL_ACT [CONVUNITS][5][16][2] = '{'{
        '{'{1, 224}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
        '{'{1, 112}, '{115, 226}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
        '{'{1, 56}, '{59, 114}, '{117, 172}, '{175, 230}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
        '{'{1, 28}, '{31, 58}, '{61, 88}, '{91, 118}, '{121, 148}, '{151, 178}, '{181, 208}, '{211, 238}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
        '{'{1, 14}, '{17, 30}, '{33, 46}, '{49, 62}, '{65, 78}, '{81, 94}, '{97, 110}, '{113, 126}, '{129, 142}, '{145, 158}, '{161, 174}, '{177, 190}, '{193, 206}, '{209, 222}, '{225, 238}, '{241, 254}}}};
    localparam int PARALLEL_KER [CONVUNITS][5][16][2] = '{'{
        '{'{0, 111}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
        '{'{0, 55}, '{57, 112}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
        '{'{0, 27}, '{29, 56}, '{58, 85}, '{87, 114}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
        '{'{0, 13}, '{15, 28}, '{30, 43}, '{45, 58}, '{60, 73}, '{75, 88}, '{90, 103}, '{105, 118}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}, '{0, 0}},
        '{'{0, 6}, '{8, 14}, '{16, 22}, '{24, 30}, '{32, 38}, '{40, 46}, '{48, 54}, '{56, 62}, '{64, 70}, '{72, 78}, '{80, 86}, '{88, 94}, '{96, 102}, '{104, 110}, '{112, 118}, '{120, 126}}}};

    localparam int STRIDE_DIM [CONVUNITS] = '{2};
    localparam int STRIDE [CONVUNITS][2]  = '{'{1, 2}};
    localparam int STRIDE_MAX [CONVUNITS] = '{2};

endpackage


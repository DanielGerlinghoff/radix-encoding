`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 19/04/2021
//
// Description: Store configuration for convolution units during runtime
//
//////////////////////////////////////////////////////////////////////////////////


interface if_configuration;
    import pkg_convolution::CONVUNITS, pkg_pooling::POOLUNITS, pkg_linear::LINUNITS;
    logic enable [CONVUNITS+POOLUNITS+LINUNITS];

    /* Convolution settings */
    import pkg_convolution::CONV_BITS;
    localparam CONV_PARALLEL_BITS = $clog2($size(pkg_convolution::PARALLEL_NUM, 2) + 1);
    localparam CONV_STRIDE_BITS   = $clog2($size(pkg_convolution::STRIDE, 2) + 1);

    typedef enum logic [0:1] {
        DIR = 2'b10,
        ADD = 2'b01,
        SFT = 2'b00,
        DEL = 2'b11
    } output_modes;

    logic [CONV_PARALLEL_BITS-1:0] conv_parallel;
    logic [CONV_STRIDE_BITS-1:0]   conv_stride;
    logic                          conv_padding;
    output_modes                   output_mode;
    logic [$clog2(CONV_BITS)-1:0]  act_scale;

    /* Pooling settings */
    localparam POOL_PARALLEL_BITS = $clog2($size(pkg_pooling::PARALLEL_NUM, 2) + 1);

    logic [POOL_PARALLEL_BITS-1:0] pool_parallel;

    /* Linear settings */
    logic [$clog2(pkg_linear::CHANNELS_MAX+1)-1:0] lin_channels;
    logic lin_relu;

endinterface

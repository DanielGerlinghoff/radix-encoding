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
    logic [$clog2(pkg_linear::LIN_CHANNELS_MAX+1)-1:0] lin_channels;
    logic lin_relu;

    /* Modports */
    modport proc (
        output enable,
        output conv_parallel,
        output conv_stride,
        output conv_padding,
        output pool_parallel,
        output lin_channels,
        output lin_relu,
        output output_mode,
        output act_scale
    );

    modport conv_in (
        input enable,
        input conv_parallel,
        input conv_stride
    );

    modport conv_array (
        input enable,
        input conv_parallel,
        input conv_padding
    );

    modport conv_out (
        input enable,
        input output_mode
    );

    modport conv_relu (
        input conv_parallel,
        input act_scale
    );

    modport pool_in (
        input enable,
        input pool_parallel
    );

    modport pool_array (
        input enable
    );

    modport pool_out (
        input pool_parallel,
        input output_mode
    );

    modport lin (
        input enable,
        input lin_channels,
        input lin_relu,
        input act_scale
    );

endinterface

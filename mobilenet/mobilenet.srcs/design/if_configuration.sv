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

    import pkg_processing::CONVUNITS, pkg_processing::CONV_BITS;
    localparam PARALLEL_BITS = $clog2($size(pkg_processing::PARALLEL_NUM, 2) + 1);
    localparam STRIDE_BITS   = $clog2($size(pkg_processing::STRIDE, 2) + 1);

    typedef enum logic [0:1] {
        DIR = 2'b10,
        ADD = 2'b01,
        SFT = 2'b00,
        DEL = 2'b11
    } output_modes;

    logic                         enable [CONVUNITS];
    logic [PARALLEL_BITS-1:0]     conv_parallel;
    logic [STRIDE_BITS-1:0]       conv_stride;
    logic                         conv_padding;
    output_modes                  output_mode;
    logic [$clog2(CONV_BITS)-1:0] act_scale;

    import pkg_pooling::POOLUNITS;
    logic pool_enable [POOLUNITS];
    logic [2-1:0] pool_parallel;

    /* Modports */
    modport proc (
        output enable,
        output conv_parallel,
        output conv_stride,
        output conv_padding,
        output output_mode,
        output act_scale
    );

    modport array_in (
        input enable,
        input conv_parallel,
        input conv_stride
    );

    modport array (
        input enable,
        input conv_parallel,
        input conv_padding
    );

    modport array_out (
        input enable,
        input output_mode
    );

    modport array_relu (
        input conv_parallel,
        input act_scale
    );

    modport pool_in (
        input pool_enable,
        input pool_parallel
    );

    modport pool_array (
        input pool_enable
    );

    modport pool_out (
        input pool_parallel
    );

endinterface

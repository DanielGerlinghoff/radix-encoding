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

    import pkg_processing::*;

    localparam PARALLEL_BITS = $clog2($size(PARALLEL_NUM, 2) + 1);
    localparam STRIDE_BITS   = $clog2($size(STRIDE, 2) + 1);

    typedef enum logic [0:1] {
        DIR = 2'b10,
        ADD = 2'b01,
        SFT = 2'b00,
        DEL = 2'b11
    } output_modes;

    logic                         enable [CONVUNITS];
    logic [PARALLEL_BITS-1:0]     conv_parallel;
    logic [STRIDE_BITS-1:0]       conv_stride;
    output_modes                  output_mode;
    logic [$clog2(CONV_BITS)-1:0] act_scale;

    /* Modports */
    modport proc (
        output enable,
        output conv_parallel,
        output conv_stride,
        output output_mode,
        output act_scale
    );

    modport array_in (
        input enable,
        input conv_parallel,
        input conv_stride
    );

    modport array_out (
        input enable,
        input output_mode
    );

    modport array_relu (
        input conv_parallel,
        input act_scale
    );

endinterface

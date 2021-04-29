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

    logic                     enable [CONVUNITS];
    logic [PARALLEL_BITS-1:0] conv_parallel;
    logic [STRIDE_BITS-1:0]   conv_stride;

    enum logic [0:1] {
        DIR = 2'b10,
        ADD = 2'b01,
        SFT = 2'b00
    } output_mode;

    /* Modports */
    modport array_in (
        input enable,
        input conv_parallel,
        input conv_stride
    );

    modport array_out (
        input enable,
        input output_mode
    );

endinterface

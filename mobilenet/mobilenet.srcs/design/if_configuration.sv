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
    import pkg_memory::ACT_NUM;

    localparam PARALLEL_BITS = $clog2($size(PARALLEL_NUM, 2) + 1);
    localparam STRIDE_BITS   = $clog2($size(STRIDE, 2) + 1);

    logic                       enable [CONVUNITS];
    logic [PARALLEL_BITS-1:0]   conv_parallel;
    logic [STRIDE_BITS-1:0]     conv_stride;
    logic [$clog2(ACT_NUM)-1:0] mem_select;

    enum logic [0:1] {
        DIR = 2'b10,
        ADD = 2'b01,
        SFT = 2'b00
    } output_mode;

    /* Modports */
    modport array_in (
        input enable,
        input mem_select,
        input conv_parallel,
        input conv_stride
    );

    modport array_out (
        input enable,
        input output_mode
    );

    modport array_relu (
        input conv_parallel,
        input mem_select
    );

endinterface

`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 19/04/2021
// 
// Description: Store following configuration data for convolution arrays:
//
//              Enable = 1'b0: Disable conv. array
//                     = 1'b1: Enable
//
//              Stride: Values stored in pkg_configuration::STRIDE
//
//              Parallel: Values stored in pkg_configuration::PARALLEL
//
//////////////////////////////////////////////////////////////////////////////////


interface if_configuration;

    import pkg_configuration::*;

    localparam PARALLEL_BITS = $clog2(PARALLEL_DIM[0]);
    localparam STRIDE_BITS   = $clog2(STRIDE_DIM);

    logic                     enable [CONVUNITS];
    logic [PARALLEL_BITS-1:0] parallel [CONVUNITS];
    logic [STRIDE_BITS-1:0]   stride [CONVUNITS];

    enum logic [0:1] {
        DIR = 2'b10,
        ADD = 2'b01,
        SFT = 2'b00
    } output_mode;

    /* Modports */
    modport array_in (
        input enable,
        input parallel,
        input stride
    );

    modport array_out (
        input enable,
        input output_mode
    );

endinterface

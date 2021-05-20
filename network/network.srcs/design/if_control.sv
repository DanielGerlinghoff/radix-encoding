`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 04/05/2021
//
// Description: Distribute control signals to processing units
//
//////////////////////////////////////////////////////////////////////////////////


interface if_control;

    import pkg_convolution::CONVUNITS;
    import pkg_pooling::POOLUNITS;

    logic reset;
    logic start;
    logic finish [CONVUNITS+POOLUNITS];

endinterface


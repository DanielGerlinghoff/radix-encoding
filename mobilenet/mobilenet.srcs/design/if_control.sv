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

    logic reset;
    logic start;
    logic finish [pkg_processing::CONVUNITS];

    /* Modports */
    modport array (
        input  reset,
        input  start,
        output finish
    );

    modport pool (
        input  reset,
        input  start,
        output finish
    );

    modport proc (
        output reset,
        output start,
        input  finish
    );

endinterface


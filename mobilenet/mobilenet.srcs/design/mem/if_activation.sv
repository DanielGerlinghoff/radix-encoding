`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 05/04/2021
// 
// Description: Connect activation BRAMs to convolution arrays
// 
//////////////////////////////////////////////////////////////////////////////////


interface if_activation #(
    SIZE_MAX
);

    logic [SIZE_MAX-1:0] data;
    logic wren;

    /* Modports */
    modport array_in (
        input data,
        input wren
    );

endinterface


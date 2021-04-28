`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 01/04/2021
// 
// Description: Connect kernel BRAMs to convolution arrays
// 
//////////////////////////////////////////////////////////////////////////////////


interface if_kernel #(
    KER_BITS,
    KER_VALS
);

    logic [KER_VALS-1:0][KER_BITS-1:0] data;
    logic                              wren;
    
    /* Modports */
    modport array (
        input data,
        input wren
    );

    modport mem (
    );


endinterface


`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 29/04/2021
//
// Description: Combine BRAMs for kernel & activations and processing units
//
//////////////////////////////////////////////////////////////////////////////////


module network (
    input  logic                                  clk,
    output logic                                  dram_enable,
    output logic [pkg_memory::DRAM_ADDR_BITS-1:0] dram_addr,
    input  logic [pkg_memory::DRAM_DATA_BITS-1:0] dram_data
);

    import pkg_processing::CONVUNITS;

    /* Interfaces */
    if_configuration conf ();
    if_activation act (.clk);
    if_kernel ker (.clk);

    logic reset = 0;
    logic start = 0;
    logic finish [CONVUNITS];

    /* Processing units */
    generate
        for (genvar cu = 0; cu < CONVUNITS; cu++) begin :gen_convunits
            conv_unit #(
                .ID (cu)
            ) conv_unit_i (
                .clk,
                .conf,
                .ker,
                .act,
                .rst    (reset),
                .start  (start),
                .finish (finish[cu])
            );
        end
    endgenerate

    /* Memory units */
    mem_kernel mem_ker (
        .clk,
        .ker
    );

    mem_activation mem_act (
        .clk,
        .act
    );

endmodule


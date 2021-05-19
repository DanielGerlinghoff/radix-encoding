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
    input  logic                                                             clk,
    input  logic                                                             proc_reset,
    input  logic                                                             proc_start,
    output logic                                                             proc_finish,
    input  logic                                                             input_en,
    input  logic [$clog2(pkg_memory::ACT_HEIGHT_MAX)-1:0]                    input_addr,
    input  logic [0:pkg_memory::ACT_WIDTH_MAX-1]                             input_data,
    input  logic                                                             output_en,
    input  logic [$clog2(pkg_memory::ACT_HEIGHT[pkg_memory::ACT_NUM-1])-1:0] output_addr,
    output logic [0:pkg_memory::ACT_WIDTH[pkg_memory::ACT_NUM-1]-1]          output_data,
    output logic                                                             dram_enable,
    output logic [pkg_memory::DRAM_ADDR_BITS-1:0]                            dram_addr,
    input  logic [pkg_memory::DRAM_DATA_BITS-1:0]                            dram_data
);

    /* Interfaces */
    if_configuration conf ();
    if_control ctrl ();
    if_activation act (.clk);
    if_kernel ker (.clk);

    assign act.in_en    = input_en;
    assign act.in_addr  = input_addr;
    assign act.in_data  = input_data;
    assign act.out_en   = output_en;
    assign act.out_addr = output_addr;
    assign output_data  = act.out_data;

    /* Control unit */
    processor proc (
        .clk,
        .conf, .ctrl, .ker, .act,
        .reset  (proc_reset),
        .start  (proc_start),
        .finish (proc_finish)
    );

    /* Processing units */
    import pkg_convolution::CONVUNITS, pkg_pooling::POOLUNITS, pkg_linear::LINUNITS;

    generate
        for (genvar cu = 0; cu < CONVUNITS; cu++) begin :gen_convunits
            conv_unit #(
                .ID (cu)
            ) conv_unit_i (
                .clk,
                .conf, .ctrl, .ker, .act
            );
        end
    endgenerate

    generate
        for (genvar pu = 0; pu < POOLUNITS; pu++) begin :gen_poolunits
            if (pkg_pooling::MAX_N_AVG) begin
                pool_max_unit #(
                    .ID (pu + CONVUNITS)
                ) pool_unit_i (
                    .clk,
                    .conf, .ctrl, .act
                );
            end else begin
                // TODO: Average pooling unit
            end
        end
    endgenerate

    generate
        for (genvar lu = 0; lu < LINUNITS; lu++) begin :gen_linunits
            lin_unit #(
                .ID (lu + CONVUNITS + POOLUNITS)
            ) lin_unit_i (
                .clk,
                .conf, .ctrl, .ker, .act
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


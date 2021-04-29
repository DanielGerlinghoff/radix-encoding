`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 29/04/2021
//
// Description: Contrainer for all BRAMs which store kernel values
//
//////////////////////////////////////////////////////////////////////////////////


module mem_kernel (
    if_kernel.bram ker,
    input clk
);

    import pkg_memory::*;

    generate
        for (genvar n = 0; n < KER_NUM; n++) begin :gen_bram
            localparam WR_HEIGHT = KER_HEIGHT[n] * KER_WIDTH[n] / DRAM_WIDTH;

            bram_kernel #(
                .RD_WIDTH  (KER_WIDTH[n]),
                .RD_HEIGHT (KER_HEIGHT[n]),
                .WR_WIDTH  (DRAM_WIDTH),
                .WR_HEIGHT (WR_HEIGHT),
                .INIT_FILE (KER_INIT[n])
            ) bram_kernel_i (
                .clk     (clk),
                .wr_en   (ker.bram_wr_en[n]),
                .wr_addr (ker.bram_wr_addr[$clog2(WR_HEIGHT)-1:0]),
                .wr_data (ker.bram_wr_data),
                .rd_en   (ker.bram_rd_en[n]),
                .rd_addr (ker.bram_rd_addr[$clog2(KER_HEIGHT[n])-1:0]),
                .rd_data (ker.bram_rd_data[n])
            );

        end
    endgenerate

endmodule


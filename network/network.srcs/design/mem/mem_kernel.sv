`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 29/04/2021
//
// Description: Container for all BRAMs which store kernel values
//
//////////////////////////////////////////////////////////////////////////////////


module mem_kernel (
    if_kernel.bram ker,
    input clk
);

    import pkg_memory::*;

    generate
        for (genvar n = 0; n < KER_NUM; n++) begin :gen_conv_bram
            localparam int RATIO = KER_HEIGHT[n] * KER_WIDTH[n] / DRAM_DATA_BITS;
            localparam int WR_HEIGHT = RATIO < 2 ? 2 : RATIO;

            bram_kernel #(
                .RD_WIDTH  (KER_WIDTH[n]),
                .RD_HEIGHT (KER_HEIGHT[n]),
                .WR_WIDTH  (DRAM_DATA_BITS),
                .WR_HEIGHT (WR_HEIGHT),
                .INIT_FILE (KER_INIT[n])
            ) bram_i (
                .clk     (clk),
                .wr_en   (ker.ker_bram_wr_en[n]),
                .wr_addr (ker.ker_bram_wr_addr[$clog2(WR_HEIGHT)-1:0]),
                .wr_data (ker.ker_bram_wr_data),
                .rd_en   (ker.ker_bram_rd_en[n]),
                .rd_addr (ker.ker_bram_rd_addr[$clog2(KER_HEIGHT[n])-1:0]),
                .rd_data (ker.ker_bram_rd_data[n])
            );

        end
    endgenerate

    generate
        for (genvar n = 0; n < WGT_NUM; n++) begin :gen_lin_bram
            bram_kernel #(
                .RD_WIDTH  (DRAM_DATA_BITS),
                .RD_HEIGHT (WGT_HEIGHT[n]),
                .WR_WIDTH  (DRAM_DATA_BITS),
                .WR_HEIGHT (WGT_HEIGHT),
                .INIT_FILE (WGT_INIT[n])
            ) bram_i (
                .clk     (clk),
                .wr_en   (ker.wgt_bram_wr_en[n]),
                .wr_addr (ker.wgt_bram_wr_addr[$clog2(WGT_HEIGHT[n])-1:0]),
                .wr_data (ker.wgt_bram_wr_data),
                .rd_en   (ker.wgt_bram_rd_en[n]),
                .rd_addr (ker.wgt_bram_rd_addr[$clog2(WGT_HEIGHT[n])-1:0]),
                .rd_data (ker.wgt_bram_rd_data[n])
            );
        end
    endgenerate

endmodule


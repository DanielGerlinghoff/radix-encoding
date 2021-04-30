`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 30/04/2021
//
// Description: Container for all BRAMs which store activation values
//
//////////////////////////////////////////////////////////////////////////////////

module mem_activation (
    if_activation.bram act,
    input logic clk
);

    import pkg_memory::*;

    generate
        for (genvar n = 0; n < ACT_NUM; n++) begin :gen_bram
            bram_activation #(
                .WIDTH  (ACT_WIDTH[n]),
                .HEIGHT (ACT_HEIGHT[n])
            ) bram_i (
                .clk     (clk),
                .addr    (act.addr),
                .wr_en   (act.wr_en[n]),
                .wr_data (act.wr_data),
                .rd_en   (act.rd_en[n]),
                .rd_data (act.rd_data[n])
            );
        end
    endgenerate

endmodule


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
    if_activation act,
    input logic clk
);

    import pkg_memory::*;

    generate
        for (genvar n = 0; n < ACT_NUM; n++) begin :gen_bram
            logic [0:ACT_WIDTH[n]-1] act_rd_data, act_wr_data;
            assign act.rd_data[n] = {act_rd_data, {ACT_WIDTH_MAX-ACT_WIDTH[n] {1'b0}}};
            assign act_wr_data = act.wr_data[0:ACT_WIDTH[n]-1];

            if (n != ACT_NUM - 1) begin
                bram_activation #(
                    .WIDTH     (ACT_WIDTH[n]),
                    .HEIGHT    (ACT_HEIGHT[n]),
`ifndef SYNTHESIS   .INIT_FILE (n == 0 ? ACT_INIT : "")
`else               .INIT_FILE ("") `endif
                ) bram_i (
                    .clk     (clk),
                    .wr_en   (act.wr_en[n]),
                    .wr_addr (act.wr_addr),
                    .wr_data (act_wr_data),
                    .rd_en   (act.rd_en[n]),
                    .rd_addr (act.rd_addr),
                    .rd_data (act_rd_data)
                );
            end else begin
                bram_activation #(
                    .WIDTH     (ACT_WIDTH[n]),
                    .HEIGHT    (ACT_HEIGHT[n]),
`ifndef SYNTHESIS   .INIT_FILE (n == 0 ? ACT_INIT : "")
`else               .INIT_FILE ("") `endif
                ) bram_i (
                    .clk     (clk),
                    .wr_en   (act.wr_en[n]),
                    .wr_addr (act.wr_addr),
                    .wr_data (act_wr_data),
                    .rd_en   (act.out_en),
                    .rd_addr (act.out_addr),
                    .rd_data (act.out_data)
                );
            end
        end
    endgenerate

endmodule


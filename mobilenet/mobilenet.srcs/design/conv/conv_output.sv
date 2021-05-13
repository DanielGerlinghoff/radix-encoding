`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 22/03/2021
//
// Description: Read convoluted rows from array and store them into intermediate
//              BRAM depending on the configured output mode
//
//////////////////////////////////////////////////////////////////////////////////


module conv_output
import pkg_convolution::*;
#(
    ID
) (
    if_configuration.conv_out conf,
    input  logic                               clk, rst,
    input  logic [CONV_BITS-1:0]               act_row [CONV_SIZE[ID]],
    input  logic                               act_valid,

    output logic [$clog2(CONV_SIZE[ID])-1:0]   bram_addr,
    input  logic [CONV_SIZE[ID]*CONV_BITS-1:0] bram_rd_data,
    output logic [CONV_SIZE[ID]*CONV_BITS-1:0] bram_wr_data,
    output logic                               bram_wr_en
);

    localparam SIZE = CONV_SIZE[ID];

    /* BRAM control */
    logic [$clog2(SIZE)-1:0] act_cnt;
    logic                    act_add = 0;
    logic [CONV_BITS-1:0]    act_reg [SIZE];
    logic [CONV_BITS-1:0]    act_data_new [SIZE], act_data_old [SIZE];

    always_ff @(posedge clk) begin
        if (rst) begin
            bram_wr_en <= 0;
            act_cnt <= 0;

        end else if (conf.enable[ID] && conf.conv_output_mode != conf.DEL) begin
            if (act_valid) begin
                act_reg <= act_row;
                act_add <= 1;

            end else if (act_add) begin
                act_add <= 0;
                bram_wr_en   <= 1;
                bram_wr_data <= {>>{act_data_new}};

            end else if (bram_wr_en) begin
                bram_wr_en <= 0;
                act_cnt <= act_cnt + 1;
            end
        end
    end

    assign bram_addr          = act_cnt;
    assign {>>{act_data_old}} = bram_rd_data;

    /* Adder array */
    generate
        for (genvar s = 0; s < SIZE; s++) begin :gen_adders
            assign act_data_new[s] = act_reg[s] +
                (conf.conv_output_mode[0] ? 0 : (conf.conv_output_mode[1] ? act_data_old[s] : act_data_old[s] << 1));
        end
    endgenerate

endmodule


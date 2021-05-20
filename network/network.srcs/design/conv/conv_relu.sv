`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 30/04/2021
//
// Description: Reduce bit width and apply ReLU activation function before
//              writing values to activation ping-pong BRAM
//
//////////////////////////////////////////////////////////////////////////////////


module conv_relu
import pkg_convolution::*;
#(
    ID
) (
    if_configuration conf,
    if_activation act,
    input logic clk,
    input logic [0:CONV_SIZE[ID]-1][CONV_BITS-1:0] conv_data
);

    /* Parallel unpacking */
    logic [0:PARALLEL_DIM[ID][0]-1][0:PARALLEL_MAX[ID]-1][0:CONV_SIZE[ID]-1][CONV_BITS-1:0] conv_parallel;
    logic [CONV_BITS-1:0] conv_unpacked [PARALLEL_MAX[ID]][CONV_SIZE[ID]];
    wire  conv_unpack = act.conv_rd_val[ID];

    generate
        for (genvar p = 0; p < PARALLEL_DIM[ID][0]; p++) begin :gen_parallel
            for (genvar a = 0; a < PARALLEL_NUM[ID][p]; a++) begin :gen_parallel_assign
                localparam int pos [2] = PARALLEL_OUT[ID][p][a];
                localparam int pad = CONV_SIZE[ID] - (pos[1] - pos[0] + 1);
                assign conv_parallel[p][a] = {conv_data[pos[0]:pos[1]], {pad*CONV_BITS {1'bx}}};
            end
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (conv_unpack) begin
            {>>{conv_unpacked}} <= conv_parallel[conf.conv_parallel];
        end
    end

    /* Quantization and ReLU function */
    logic [$clog2(PARALLEL_MAX[ID])-1:0] cnt_assign;
    logic [ACT_BITS-1:0] conv_activated [CONV_SIZE[ID]];
    logic conv_activate;

    function logic [CONV_BITS-2:0] relu (input logic [CONV_BITS-1:0] conv_output);
        if (!conv_output[CONV_BITS-1]) relu = conv_output;
        else                           relu = 0;
    endfunction

    function logic [ACT_BITS-1:0] quantize (input logic [CONV_BITS-2:0] conv_relu);
        automatic logic [CONV_BITS-2:0] conv_shifted = conv_relu >> conf.act_scale;
        automatic logic rounding = conv_relu[conf.act_scale-1];

        if (|conv_shifted[CONV_BITS-2:ACT_BITS]) quantize = 2 ** ACT_BITS - 1;
        else                                     quantize = conv_shifted[ACT_BITS-1:0] + rounding;
    endfunction

    always_ff @(posedge clk) begin
        if (conv_activate) begin
            for (int val = 0; val < CONV_SIZE[ID]; val++)
                conv_activated[val] <= quantize(relu(conv_unpacked[cnt_assign][val]));
        end
    end

    /* Write to activation BRAM */
    logic                        conv_write = 0;
    logic [$clog2(ACT_BITS)-1:0] cnt_bits;
    logic [$high(act.wr_addr):0] wr_addr_offset;

    always_ff @(posedge clk) begin
        if (conv_write) begin
            act.wr_en_u[ID][act.mem_wr_select] <= 1;
            for (int val = 0; val < CONV_SIZE[ID]; val++) begin
                act.wr_data_u[ID][val] <= conv_activated[val][cnt_bits];
            end
        end else begin
            act.wr_en_u[ID] <= '{default: 0};
        end

        if (conv_activate) begin
            if (!cnt_assign) begin
                wr_addr_offset <= 0;
            end else begin
                wr_addr_offset <= wr_addr_offset - act.addr_step[1];
            end
        end else if (conv_write) begin
            wr_addr_offset <= wr_addr_offset + act.addr_step[0];
        end

        act.wr_addr_offset_u[ID] <= wr_addr_offset;
    end

    /* Process control */
    logic finish;

    always_ff @(posedge clk) begin
        if (conv_unpack) begin
            conv_activate <= 1;
            cnt_assign    <= 0;
        end

        if (conv_activate) begin
            conv_activate <= 0;
            conv_write    <= 1;
            cnt_bits      <= ACT_BITS - 1;
        end

        finish <= 0;
        if (conv_write) begin
            if (cnt_bits == 1) begin
                cnt_bits <= 0;
                if (cnt_assign != PARALLEL_NUM[ID][conf.conv_parallel] - 1) begin
                    conv_activate <= 1;
                    cnt_assign    <= cnt_assign + 1;
                end else begin
                    cnt_assign <= 0;
                    finish <= 1;
                end
            end else if (cnt_bits == 0) begin
                conv_write <= conv_activate;
            end else begin
                cnt_bits <= cnt_bits - 1;
            end
        end
    end

    assign act.transfer_finish[ID] = finish;


endmodule


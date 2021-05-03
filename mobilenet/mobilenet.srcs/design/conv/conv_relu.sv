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
import pkg_processing::*;
#(
    ID
) (
    if_configuration.array_relu conf,
    if_activation.array_relu act,
    input logic clk,
    input logic [0:CONV_SIZE[ID]-1][CONV_BITS-1:0] conv_data
);

    /* Parallel unpacking */
    tri0  [0:PARALLEL_MAX[ID]-1][0:CONV_SIZE[ID]-1][CONV_BITS-1:0] conv_parallel;
    logic [CONV_BITS-1:0] conv_unpacked [PARALLEL_MAX[ID]][CONV_SIZE[ID]];
    wire  conv_unpack = act.conv_rd_val[ID];

    generate
        for (genvar p = 0; p < PARALLEL_DIM[ID][0]; p++) begin :gen_parallel
            for (genvar a = 0; a < PARALLEL_NUM[ID][p]; a++) begin :gen_parallel_assign
                localparam int pos [2] = PARALLEL_KER[ID][p][a];
                localparam int pad = CONV_SIZE[ID] - (pos[1] - pos[0] + 1);
                assign conv_parallel[a] = (conf.conv_parallel == p) ? {conv_data[pos[0]:pos[1]], {pad*CONV_BITS {1'bx}}} : 'z;
            end
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (conv_unpack) begin
            {>>{conv_unpacked}} <= conv_parallel;;
        end
    end

    /* Quantization and ReLU function */
    logic [$clog2(PARALLEL_MAX[ID])-1:0] cnt_assign;
    logic [ACT_BITS-1:0] conv_activated [CONV_SIZE[ID]];
    logic conv_activate;

    function logic [CONV_BITS-1:0] relu (input logic [CONV_BITS-1:0] conv_output);
        if (!conv_output[CONV_BITS-1]) relu = conv_output;
        else                           relu = 0;
    endfunction

    function logic [ACT_BITS-1:0] quantize (input logic [CONV_BITS-1:0] conv_relu);
        automatic logic [CONV_BITS-1:0] conv_shifted = conv_relu >> act.conv_scale;

        if (|conv_shifted[CONV_BITS-1:ACT_BITS-1]) quantize = 2 ** (ACT_BITS - 1) - 1;
        else                                       quantize = conv_shifted[ACT_BITS-1:0];
    endfunction

    always_ff @(posedge clk) begin
        if (conv_activate)
            for (int val = 0; val < CONV_SIZE[ID]; val++) begin
                conv_activated[val] <= quantize(relu(conv_unpacked[cnt_assign][val]));
        end
    end

    /* Write to activation BRAM */
    logic [$clog2(ACT_BITS)-1:0] cnt_bits;
    logic conv_write;

    always_ff @(posedge clk) begin
        if (conv_write) begin
            act.wr_en[conf.mem_select] <= 1;
            for (int val = 0; val < CONV_SIZE[ID]; val++) begin
                act.wr_data[CONV_SIZE[ID]-val-1] <= conv_activated[val][cnt_bits];
            end
        end else begin
            act.wr_en[conf.mem_select] <= 0;
            act.wr_data       <= 'z;
        end

        if (conv_activate) begin
            if (!cnt_assign) begin
                act.wr_addr_offset <= 0;
            end else begin
                act.wr_addr_offset <= act.wr_addr_offset - act.conv_addr_step[1];
            end
        end else if (conv_write) begin
            act.wr_addr_offset <= act.wr_addr_offset + act.conv_addr_step[0];
        end else begin
            act.wr_addr_offset <= 'z;
        end

    end

    assign act.wr_add_addr = conv_write ? 1'b1 : 1'bz;

    /* Process control */
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

        if (conv_write) begin
            if (cnt_bits == 1) begin
                cnt_bits <= 0;
                if (cnt_assign != PARALLEL_NUM[ID][conf.conv_parallel] - 1) begin
                    conv_activate <= 1;
                    cnt_assign    <= cnt_assign + 1;
                end else begin
                    cnt_assign <= 0;
                    act.conv_transfer_finish <= 1;
                end
            end else if (cnt_bits == 0) begin
                conv_write <= conv_activate;
            end else begin
                cnt_bits <= cnt_bits - 1;
            end
        end

        if (act.conv_transfer_finish) begin
            act.conv_transfer_finish <= 0;
        end

    end

endmodule


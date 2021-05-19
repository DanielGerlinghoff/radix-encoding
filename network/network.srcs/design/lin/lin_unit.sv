`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 16/05/2021
//
// Description: Row of adders to accumulate kernel values of multiple output
//              channels
//
//////////////////////////////////////////////////////////////////////////////////


module lin_unit
import pkg_linear::*;
#(
    ID
) (
    if_configuration.lin conf,
    if_control.lin ctrl,
    if_kernel.lin ker,
    if_activation.lin act,
    input logic clk
);

    /* Register weights and activations */
    logic signed [WGT_BITS-1:0] weights_reg [LIN_SIZE-1:0];
    logic act_reg;

    always_ff @(posedge clk) begin
        if (conf.enable[ID]) begin
            if (ker.wgt_bram_rd_val[ker.wgt_select])
                {>>{weights_reg}} <= ker.wgt_bram_rd_data[ker.wgt_select][WGT_BITS*LIN_SIZE-1:0];

            if (act.rd_val[act.mem_rd_select])
                {>>{act_reg}} <= act.rd_data[act.mem_rd_select][0];
        end
    end

    /* Accumulate and shift */
    logic enable, activate;
    logic signed [SUM_BITS-1:0] lin_sum [LIN_SIZE];
    logic [$high(conf.lin_channels):0] cnt_chn;
    logic [$clog2(ACT_BITS)-1:0] cnt_tstep;

    always_ff @(posedge clk) begin
        enable <= act.rd_val[act.mem_rd_select] && conf.enable[ID];
        activate <= 0;

        if (ctrl.reset) begin
            lin_sum <= '{default: 0};
            cnt_chn   <= 0;
            cnt_tstep <= 0;

        end else if (enable) begin
            for (int s = 0; s < LIN_SIZE; s++) begin
                lin_sum[s] <= act_reg ? lin_sum[s] + weights_reg[s] : lin_sum[s];
            end
            cnt_chn <= cnt_chn + 1;

        end else if (cnt_chn == conf.lin_channels) begin
            cnt_chn <= 0;
            if (cnt_tstep != ACT_BITS - 1) begin
                cnt_tstep <= cnt_tstep + 1;
                for (int s = 0; s < LIN_SIZE; s++) begin
                    lin_sum[s] <= lin_sum[s] << 1;
                end
            end else begin
                cnt_tstep <= 0;
                activate <= 1;
            end
        end
    end

    /* Activation function and scaling */
    logic [ACT_BITS-1:0] lin_activated [LIN_SIZE];

    function logic [SUM_BITS-2:0] relu (input logic [SUM_BITS-1:0] lin_output);
        if (!lin_output[SUM_BITS-1]) relu = lin_output;
        else                         relu = 0;
    endfunction

    function logic [ACT_BITS-1:0] quantize (input logic [SUM_BITS-2:0] lin_relu);
        automatic logic [SUM_BITS-2:0] lin_shifted = lin_relu >> conf.act_scale;
        automatic logic rounding = lin_relu[conf.act_scale-1];

        if (|lin_shifted[SUM_BITS-2:ACT_BITS]) quantize = 2 ** ACT_BITS - 1;
        else                                   quantize = lin_shifted[ACT_BITS-1:0] + rounding;
    endfunction

    always_ff @(posedge clk) begin
        if (activate) begin
            for (int s = 0; s < LIN_SIZE; s++) begin
                lin_activated[s] <= quantize(relu(lin_sum[s]));
            end
        end
    end

    /* Write to activation BRAM */
    logic write = 0;
    logic [$clog2(LIN_SIZE)-1:0] cnt_val;
    logic [$clog2(ACT_BITS)-1:0] cnt_bits;
    logic [$high(act.wr_addr):0] wr_addr_offset;

    always_ff @(posedge clk) begin
        if (activate) begin
            wr_addr_offset <= 0;
        end else if (write) begin
            act.wr_en_u[ID][act.mem_wr_select] <= 1;
            if (conf.lin_relu) begin
                act.wr_data_u[ID][0] <= lin_activated[cnt_val][cnt_bits];
            end else begin
                act.wr_data_u[ID][0:SUM_BITS-1] <= lin_sum[cnt_val];
            end

            if (!cnt_bits) begin
                wr_addr_offset <= wr_addr_offset - act.addr_step[1];
            end else begin
                wr_addr_offset <= wr_addr_offset + act.addr_step[0];
            end
        end else begin
            act.wr_en_u[ID] <= '{default: 0};
        end

        act.wr_addr_offset_u[ID] <= wr_addr_offset;
    end

    /* Process control */
    localparam LIN_SIZE_LAST = pkg_memory::ACT_HEIGHT[pkg_memory::ACT_NUM-1];

    logic finish;

    always_ff @(posedge clk) begin
        if (activate) begin
            write <= 1;
            cnt_val  <= 0;
            cnt_bits <= ACT_BITS - 1;
        end

        finish <= 0;
        if (write) begin
            if (!conf.lin_relu) begin
                if (cnt_val != LIN_SIZE_LAST - 1) begin
                    cnt_val <= cnt_val + 1;
                    write   <= 1;
                end else begin
                    cnt_val <= 0;
                    write   <= 0;
                    finish  <= 1;
                end
            end else if (!cnt_bits) begin
                cnt_bits <= ACT_BITS - 1;
                if (cnt_val != LIN_SIZE - 1) begin
                    cnt_val <= cnt_val + 1;
                    write   <= 1;
                end else begin
                    cnt_val <= 0;
                    write   <= 0;
                    finish  <= 1;
                end
            end else begin
                cnt_bits <= cnt_bits - 1;
            end
        end
    end

    assign act.transfer_finish[ID] = finish;

endmodule


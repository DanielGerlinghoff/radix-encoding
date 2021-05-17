`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 12/05/2021
//
// Description: Read pooled rows and write individual channels back to activation
//              BRAM
//
//////////////////////////////////////////////////////////////////////////////////


module pool_max_output
import pkg_pooling::*;
#(
    ID
) (
    if_configuration.pool_out conf,
    if_activation.pool_out act,
    input  logic                clk,
    input  logic [ACT_BITS-1:0] pool_row [POOL_SIZE[ID]],
    input  logic                pool_valid
);

    /* Parallel unpacking */
    tri0  [ACT_BITS-1:0] pool_parallel [PARALLEL_MAX[ID]][POOL_SIZE[ID]];
    logic [ACT_BITS-1:0] pool_unpacked [PARALLEL_MAX[ID]][POOL_SIZE[ID]];
    wire  pool_unpack = (conf.output_mode == conf.DIR) && pool_valid;

    generate
        for (genvar p = 0; p < PARALLEL_DIM[ID][0]; p++) begin :gen_parallel
            for (genvar a = 0; a < PARALLEL_NUM[ID][p]; a++) begin :gen_parallel_assign
                localparam int pos [2] = PARALLEL_OUT[ID][p][a];
                for (genvar v = pos[0]; v <= pos[1]; v++) begin :gen_parallel_values
                    assign pool_parallel[a][v-pos[0]] = (conf.pool_parallel == p) ? pool_row[v] : 'z;
                end
            end
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (pool_unpack) begin
            pool_unpacked <= pool_parallel;
        end
    end

    /* Write to activation BRAM */
    logic                                pool_write = 0;
    logic [$clog2(PARALLEL_MAX[ID])-1:0] cnt_assign;
    logic [$clog2(ACT_BITS)-1:0]         cnt_bits;
    logic [$high(act.wr_addr_offset):0]  act_addr_offset;
    logic                                act_en [$size(act.wr_en)];
    logic [0:$high(act.wr_data)]         act_data;

    always_ff @(posedge clk) begin
        if (pool_unpack) begin
            act_addr_offset <= 0;
        end else if (pool_write) begin
            act_en[act.mem_wr_select] <= 1;
            for (int val = 0; val < POOL_SIZE[ID]; val++) begin
                act_data[val] <= pool_unpacked[cnt_assign][val][cnt_bits];
            end

            if (!cnt_bits) begin
                act_addr_offset <= act_addr_offset - act.addr_step[1];
            end else begin
                act_addr_offset <= act_addr_offset + act.addr_step[0];
            end
        end else begin
            act_en          <= '{default: 'z};
            act_data        <= 'z;
            act_addr_offset <= 'z;
        end
    end

    assign act.wr_en          = act_en;
    assign act.wr_addr_offset = act_addr_offset;
    assign act.wr_add_addr    = pool_write ? 1'b1 : 1'bz;
    assign act.wr_data        = act_data;

    /* Process control */
    logic finish;

    always_ff @(posedge clk) begin
        if (pool_unpack) begin
            pool_write <= 1;
            cnt_assign <= 0;
            cnt_bits   <= ACT_BITS - 1;
        end

        finish <= 0;
        if (pool_write) begin
            if (!cnt_bits) begin
                cnt_bits <= ACT_BITS - 1;
                if (cnt_assign != PARALLEL_NUM[ID][conf.pool_parallel] - 1) begin
                    cnt_assign <= cnt_assign + 1;
                    pool_write <= 1;
                end else begin
                    cnt_assign <= 0;
                    pool_write <= 0;
                    finish     <= 1;
                end
            end else begin
                cnt_bits <= cnt_bits - 1;
            end
        end
    end

    assign act.transfer_finish = finish ? 1'b1 : 1'bz;

endmodule


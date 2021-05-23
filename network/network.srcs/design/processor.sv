`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 03/05/2021
//
// Description: Read and execute neural network instructions
//
//////////////////////////////////////////////////////////////////////////////////


module processor (
    if_configuration conf,
    if_control ctrl,
    if_kernel ker,
    if_activation act,
    input  logic clk,
    input  logic reset,
    input  logic start,
    output logic finish
);

    import pkg_memory::*;

    /* Execution */
    localparam OP_BITS   = 4;
    localparam UNIT_BITS = 5;
    localparam VAL_BITS  = INS_WIDTH - OP_BITS - UNIT_BITS;
    localparam CONF_BITS = 5;
    localparam COND_BITS = 2;
    localparam ADDR_BITS = 8;

    typedef enum logic [OP_BITS-1:0] {
        ENA  = 0,
        CONF = 1,
        RST  = 2,
        PROC = 3,
        LIN  = 4,
        KERD = 5,
        KERL = 6,
        ACTL = 7,
        ACTS = 8,
        WAIT = 9,
        END  = 10
    } ops;
    typedef enum logic [CONF_BITS-1:0] {
        CPAR  = 0,
        CSTR  = 1,
        CPAD  = 2,
        PPAR  = 3,
        LCHN  = 4,
        LRELU = 5,
        OUTM  = 6,
        SCL   = 7,
        KSEL  = 8,
        WSEL  = 9,
        WADR  = 10,
        DADR  = 11,
        ASELR = 12,
        ASELW = 13,
        ASTPF = 14,
        ASTPB = 15,
        ASRC  = 16,
        ADST  = 17
    } confs;
    typedef enum logic [COND_BITS-1:0] {
        CONV = 0,
        CWR  = 1,
        TRAN = 2
    } conds;

    logic                              next;
    logic [$clog2(INS_HEIGHT)-1:0]     instr_pnt;
    logic [INS_WIDTH-1:0]              instr;
    ops                                instr_op [2];
    logic [UNIT_BITS-1:0]              instr_unit [2];
    confs                              instr_conf;
    logic [VAL_BITS-1:0]               instr_val;
    conds                              wait_cond [2];
    logic [$clog2(WGT_HEIGHT_MAX)-1:0] lin_cnt;

    assign instr_op[0]   = ops'(instr[INS_WIDTH-1-:OP_BITS]);
    assign instr_conf    = confs'(instr[INS_WIDTH-OP_BITS-1-:CONF_BITS]);
    assign instr_unit[0] = instr[INS_WIDTH-OP_BITS-1-:UNIT_BITS];
    assign instr_val     = instr[VAL_BITS-1:0];
    assign wait_cond[0]  = conds'(instr_val[VAL_BITS-1-:COND_BITS]);

    always_ff @(posedge clk) begin
        if (reset) begin
            instr_pnt <= 0;
            next <= 0;

            conf.enable        <= '{default: 0};
            ctrl.reset         <= 0;
            ctrl.start         <= 0;
            ker.ker_bram_rd_en <= '{default: 0};
            ker.wgt_bram_rd_en <= '{default: 0};
            act.rd_en          <= '{default: 0};
            act.conv_rd_en     <= '{default: 0};
            act.wr_addr_base   <= 0;
            finish             <= 0;

        end else if (start || next) begin
            case (instr_op[0])
                ENA: begin
                    conf.enable[instr_unit[0]] <= instr_val;
                end

                CONF: begin
                    case (instr_conf)
                        CPAR:  conf.conv_parallel   <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0];
                        CSTR:  conf.conv_stride     <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0];
                        CPAD:  conf.conv_padding    <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0];
                        PPAR:  conf.pool_parallel   <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0];
                        LCHN:  conf.lin_channels    <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0];
                        LRELU: conf.lin_relu        <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0];
`ifndef SYNTHESIS       OUTM:  conf.output_mode     <= conf.output_modes'(instr_val[INS_WIDTH-OP_BITS-CONF_BITS-1:0]);
`else                   OUTM:  conf.output_mode     <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0]; `endif
                        SCL:   conf.act_scale       <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0];
                        KSEL:  {ker.ker_select, ker.ker_n_wgt} <= {instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0], 1'b1};
                        WSEL:  {ker.wgt_select, ker.ker_n_wgt} <= {instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0], 1'b0};
                        WADR:  ker.wgt_bram_rd_addr <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0];
                        DADR:  ker.dram_addr_base   <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0];
                        ASELR: act.mem_rd_select    <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0];
                        ASELW: act.mem_wr_select    <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0];
                        ASTPF: act.addr_step[0]     <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0];
                        ASTPB: act.addr_step[1]     <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0];
                        ASRC:  act.rd_addr          <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0];
                        ADST:  act.wr_addr_base     <= instr[INS_WIDTH-OP_BITS-CONF_BITS-1:0];
                    endcase
                    next <= 1;
                end

                RST: begin
                    ctrl.reset <= 1;
                    next <= 0;
                end

                PROC: begin
                    ctrl.start <= 1;
                    next <= 0;
                end

                LIN: begin
                    act.rd_en[act.mem_rd_select] <= 1;
                    ker.wgt_bram_rd_en[ker.wgt_select] <= 1;
                    lin_cnt <= 0;
                    next <= 0;
                end

                KERD: begin
                    ker.dram_start <= 1;
                    ker.dram_cnt <= instr_val;
                    next <= 0;
                end

                KERL: begin
                    ker.ker_bram_rd_en[instr_unit[0]] <= 1;
                    ker.ker_bram_rd_addr <= instr_val;
                    next <= 0;
                end

                ACTL: begin
                    act.rd_en[instr_unit[0]] <= 1;
                    act.rd_addr <= instr_val;
                    next <= 0;
                end

                ACTS: begin
                    act.conv_rd_en[instr_unit[0]] <= 1;
                    act.conv_rd_addr <= instr_val[ADDR_BITS-1:0];
                    act.wr_addr_base <= instr_val[VAL_BITS-1:ADDR_BITS];
                    next <= 0;
                end

                WAIT: begin
                    next <= 0;
                end

                END: begin
                    finish <= 1;
                    next <= 0;
                end
            endcase

            instr_pnt     <= instr_pnt + 1;
            instr_op[1]   <= instr_op[0];
            instr_unit[1] <= instr_unit[0];
            wait_cond[1]  <= wait_cond[0];

        end else begin
            case (instr_op[1])
                RST: begin
                    ctrl.reset <= 0;
                    next <= 1;
                end

                PROC: begin
                    ctrl.start <= 0;
                    next <= 1;
                end

                LIN: begin
                    if (lin_cnt != conf.lin_channels - 1) begin
                        lin_cnt <= lin_cnt + 1;
                        act.rd_addr          <= act.rd_addr + 1;
                        ker.wgt_bram_rd_addr <= ker.wgt_bram_rd_addr + 1;
                    end else begin
                        act.rd_en[act.mem_rd_select] <= 0;
                        ker.wgt_bram_rd_en[ker.wgt_select] <= 0;
                        lin_cnt <= 0;
                        next <= 1;
                    end
                end

                KERD: begin
                    ker.dram_start <= 0;
                    if (ker.dram_val[1]) next <= 1;
                end

                KERL: begin
                    ker.ker_bram_rd_en[instr_unit[1]] <= 0;
                    next <= 1;
                end

                ACTL: begin
                    act.rd_en[instr_unit[1]] <= 0;
                    next <= 1;
                end

                ACTS: begin
                    act.conv_rd_en[instr_unit[1]] <= 0;
                    next <= 1;
                end

                WAIT: begin
                    case (wait_cond[1])
                        CONV: if (ctrl.finish[instr_unit[1]]) next <= 1;
                        CWR:  if (act.conv_wr_en[instr_unit[1]]) next <= 1;
                        TRAN: if (|($size(act.transfer_finish))'({>>{act.transfer_finish}})) next <= 1;
                    endcase
                end

                END: begin
                    finish <= 0;
                    instr_pnt <= 0;
                end
            endcase
        end
    end

    /* Instruction bram */
    bram_instruction #(
        .WIDTH     (INS_WIDTH),
        .HEIGHT    (INS_HEIGHT),
        .INIT_FILE (INS_INIT)
    ) bram (
        .clk  (clk),
        .addr (instr_pnt + (start | next)),
        .data (instr)
    );

endmodule


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
    localparam CONF_BITS = 4;
    localparam COND_BITS = 2;
    localparam ADDR_BITS = 8;

    typedef enum logic [OP_BITS-1:0] {
        CONF = 1,
        RST  = 2,
        PROC = 3,
        KERL = 4,
        ACTL = 5,
        ACTS = 6,
        WAIT = 7,
        END  = 8,
        LIN  = 9
    } ops;
    typedef enum logic [CONF_BITS-1:0] {
        CPAR = 0,
        STR  = 1,
        PAD  = 2,
        OUT  = 3,
        KSEL = 4,
        ASEL = 5,
        SCL  = 6,
        ASTF = 7,
        ASTB = 8,
        PPAR = 9,
        WSEL = 10,
        LCHN = 11,
        WADR = 12,
        ASRC = 13,
        ADST = 14,
        RELU = 15
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
    assign instr_unit[0] = instr[INS_WIDTH-OP_BITS-1-:UNIT_BITS];
    assign instr_val     = instr[VAL_BITS-1:0];
    assign instr_conf    = confs'(instr_val[VAL_BITS-1-:CONF_BITS]);
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
                CONF: begin
                    if (instr_unit[0] == '1) begin
                        case (instr_conf)
                            CPAR: conf.conv_parallel   <= instr_val[VAL_BITS-CONF_BITS-1:0];
                            STR:  conf.conv_stride     <= instr_val[VAL_BITS-CONF_BITS-1:0];
                            PAD:  conf.conv_padding    <= instr_val[VAL_BITS-CONF_BITS-1:0];
                            OUT:  conf.output_mode     <= conf.output_modes'(instr_val[VAL_BITS-CONF_BITS-1:0]);
                            LCHN: conf.lin_channels    <= instr_val[VAL_BITS-CONF_BITS-1:0];
                            WADR: ker.wgt_bram_rd_addr <= instr_val[VAL_BITS-CONF_BITS-1:0];
                            ASEL: {act.mem_rd_select, act.mem_wr_select} <= instr_val[VAL_BITS-CONF_BITS-1:0];
                            KSEL: ker.ker_select       <= instr_val[VAL_BITS-CONF_BITS-1:0];
                            WSEL: ker.wgt_select       <= instr_val[VAL_BITS-CONF_BITS-1:0];
                            SCL:  conf.act_scale       <= instr_val[VAL_BITS-CONF_BITS-1:0];
                            ASTF: act.addr_step[0]     <= instr_val[VAL_BITS-CONF_BITS-1:0];
                            ASTB: act.addr_step[1]     <= instr_val[VAL_BITS-CONF_BITS-1:0];
                            PPAR: conf.pool_parallel   <= instr_val[VAL_BITS-CONF_BITS-1:0];
                            ASRC: act.rd_addr          <= instr_val[VAL_BITS-CONF_BITS-1:0];
                            ADST: act.wr_addr_base     <= instr_val[VAL_BITS-CONF_BITS-1:0];
                            RELU: conf.lin_relu        <= instr_val[VAL_BITS-CONF_BITS-1:0];
                        endcase
                    end else begin
                        conf.enable[instr_unit[0]] <= instr_val;
                    end
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
                        TRAN: if (|act.transfer_finish) next <= 1;
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


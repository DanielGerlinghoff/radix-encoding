`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 16/04/2021
// 
// Description: Receives and sends bytes using the UART interface
// 
//////////////////////////////////////////////////////////////////////////////////


module uart #(
    BITWIDTH  = 8,
    BAUDRATE  = 921600,
    CLKPERIOD = 4.284e-9,
    STOPBITS  = 1
)(
    input  logic                clk,
    input  logic                rxd, cts,
    output logic                txd, rts,
    inout  logic                rstn,

    input  logic [BITWIDTH-1:0] tx_data,
    input  logic                tx_en,
    output logic                tx_rdy_n,
    output logic [BITWIDTH-1:0] rx_data,
    output logic                rx_valid
);

    localparam int BITCOUNT = 1 / (BAUDRATE * CLKPERIOD);
    localparam     CLKCNT_WIDTH = $clog2(BITCOUNT) + 1;
    localparam     BITCNT_WIDTH = $clog2(BITWIDTH);

    /* Receive and send */
    logic [BITCNT_WIDTH-1:0] recv_bit = 0;
    logic [CLKCNT_WIDTH-1:0] recv_cnt = 0;
    logic [BITWIDTH-1:0]     recv_data;

    logic [BITCNT_WIDTH-1:0] send_bit = 0;
    logic [CLKCNT_WIDTH-1:0] send_cnt = 0;
    logic [BITWIDTH-1:0]     send_data;
    logic                    send_rd_en;
    logic                    send_empty;

    enum logic [3:0] {
        IDLE  = 4'b0001,
        START = 4'b0010,
        DATA  = 4'b0100,
        STOP  = 4'b1000
    } recv_state = IDLE, send_state = IDLE;

    always_ff @(posedge clk) begin
        rx_valid <= 0;

        unique case (recv_state)
            IDLE: begin
                if (!rxd) begin
                    recv_state <= START;
                end
            end

            START: begin
                if (recv_cnt !=  BITCOUNT / 2) begin
                    recv_cnt <= recv_cnt + 1;
                end else begin
                    recv_cnt <= 0;
                    recv_state <= DATA;
                end
            end

            DATA: begin
                if (recv_cnt != BITCOUNT) begin
                    recv_cnt <= recv_cnt + 1;
                end else begin
                    recv_cnt <= 0;
                    recv_data[recv_bit] <= rxd;

                    if (recv_bit != BITWIDTH - 1) begin
                        recv_bit <= recv_bit + 1;
                    end else begin
                        recv_bit <= 0;
                        recv_state <= STOP;
                    end
                end
            end

            STOP: begin
                if (recv_cnt != BITCOUNT * 3 / 2) begin
                    recv_cnt <= recv_cnt + 1;
                end else begin
                    recv_cnt <= 0;
                    rx_data  <= recv_data;
                    rx_valid <= 1;
                    recv_state <= IDLE;
                end
            end
            
            default begin
                recv_state <= IDLE;
            end
        endcase
    end

    always_ff @(posedge clk) begin
        send_rd_en <= 0;

        unique case (send_state)
            IDLE: begin
                txd <= 1;

                if (!send_empty) begin
                    send_rd_en <= 1;
                    send_state <= START;
                end
            end

            START: begin
                txd <= 0;

                if (send_cnt != BITCOUNT) begin
                    send_cnt <= send_cnt + 1;
                end else begin
                    send_cnt <= 0;
                    send_state <= DATA;
                end
            end

            DATA: begin
                txd <= send_data[send_bit];

                if (send_cnt != BITCOUNT) begin
                    send_cnt <= send_cnt + 1;
                end else begin
                    send_cnt <= 0;

                    if (send_bit != BITWIDTH - 1) begin
                        send_bit <= send_bit + 1;
                    end else begin
                        send_bit <= 0;
                        send_state <= STOP;
                    end
                end
            end

            STOP: begin
                txd <= 1;

                if (send_cnt != BITCOUNT * STOPBITS) begin
                    send_cnt <= send_cnt + 1;
                end else begin
                    send_cnt <= 0;
                    send_state <= IDLE;
                end
            end

            default begin
                send_state <= IDLE;
            end
        endcase
    end

    fifo_uart fifo (
        .clk         (clk),
        .din         (tx_data),
        .wr_en       (tx_en),
        .almost_full (tx_rdy_n),
        .full        (),
        .dout        (send_data),
        .rd_en       (send_rd_en),
        .empty       (send_empty)
    );

`ifdef DEBUG
    /* Logic analyzer */
    ila_uart ila (
        .clk    (clk),
        .probe0 (rxd),
        .probe1 (recv_state),
        .probe2 (rx_data),
        .probe3 (txd),
        .probe4 (send_state),
        .probe5 (tx_data)
    );
`endif

    /* Unused signals */
    assign rstn = 'z;
    assign rts  = 0;

endmodule


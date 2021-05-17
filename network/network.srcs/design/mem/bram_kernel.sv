`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 28/04/2021
//
// Description: Assymetric dual port BRAM writing data from DRAM and reading it
//              to convolution units with the respective kernel size
//
//////////////////////////////////////////////////////////////////////////////////


module bram_kernel #(
    RD_WIDTH,
    RD_HEIGHT,
    WR_WIDTH,
    WR_HEIGHT,
    INIT_FILE = ""
) (
    input  logic                         clk,
    input  logic                         wr_en,
    input  logic [$clog2(WR_HEIGHT)-1:0] wr_addr,
    input  logic [WR_WIDTH-1:0]          wr_data,
    input  logic                         rd_en,
    input  logic [$clog2(RD_HEIGHT)-1:0] rd_addr,
    output logic [RD_WIDTH-1:0]          rd_data
);

    localparam RATIO = WR_WIDTH / RD_WIDTH;

    logic [RD_WIDTH-1:0] ram [RD_HEIGHT];
    logic [$clog2(RATIO)-1:0] ls_addr;

    initial begin
        if (INIT_FILE != "") begin
            $readmemb(INIT_FILE, ram);
        end
    end

    always_ff @(posedge clk) begin
        if (rd_en) begin
            rd_data <= ram[rd_addr];
        end
    end

    always_ff @(posedge clk) begin
        if (wr_en) begin
            for (int r = 0; r < RATIO; r++) begin
                ls_addr = r;
                ram[{wr_addr, ls_addr}] <= wr_data[r*RD_WIDTH+:RD_WIDTH];
            end
        end
    end

endmodule


`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 30/04/2021
//
// Description: Simple dual port BRAM to store activations
//
//////////////////////////////////////////////////////////////////////////////////


module bram_activation #(
    WIDTH,
    HEIGHT,
    INIT_FILE = ""
) (
    input  logic                      clk,
    input  logic                      wr_en,
    input  logic [$clog2(HEIGHT)-1:0] wr_addr,
    input  logic [WIDTH-1:0]          wr_data,
    input  logic                      rd_en,
    input  logic [$clog2(HEIGHT)-1:0] rd_addr,
    output logic [WIDTH-1:0]          rd_data
);

    (* rom_style = "block" *) logic [WIDTH-1:0] ram [HEIGHT];

    initial begin
        if (INIT_FILE != "") begin
            $readmemb(INIT_FILE, ram);
        end
    end

    always_ff @(posedge clk) begin
        if (rd_en) begin
            rd_data <= ram[rd_addr];
        end

        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
    end

endmodule


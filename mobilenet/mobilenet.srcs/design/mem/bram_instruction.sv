`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 03/05/2021
//
// Description: Read-only memory to store processor instructions
//
//////////////////////////////////////////////////////////////////////////////////


module bram_instruction #(
    WIDTH,
    HEIGHT,
    INIT_FILE
) (
    input  logic                      clk,
    input  logic [$clog2(HEIGHT)-1:0] addr,
    output logic [WIDTH-1:0]          data
);

    logic [WIDTH-1:0] rom [HEIGHT];

    initial begin
        $readmemh(INIT_FILE, rom);
    end

    always_ff @(posedge clk) begin
        data <= rom[addr];
    end

endmodule


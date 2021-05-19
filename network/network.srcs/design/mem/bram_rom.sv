`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:     A*STAR IHPC
// Engineer:    Gerlinghoff Daniel
// Create Date: 18/05/2021
//
// Description: Read-only memory initialized by file
//
//////////////////////////////////////////////////////////////////////////////////


module bram_rom #(
    RD_WIDTH,
    RD_HEIGHT,
    INIT_FILE = ""
) (
    input  logic                         clk,
    input  logic                         rd_en,
    input  logic [$clog2(RD_HEIGHT)-1:0] rd_addr,
    output logic [RD_WIDTH-1:0]          rd_data
);

    (* rom_style = "block" *) logic [RD_WIDTH-1:0] rom [RD_HEIGHT];

    initial begin
        $readmemb(INIT_FILE, rom);
    end

    always_ff @(posedge clk) begin
        if (rd_en) begin
            rd_data <= rom[rd_addr];
        end
    end

endmodule


/******************************************************************************
// (c) Copyright 2013 - 2014 Xilinx, Inc. All rights reserved.
//
// This file contains confidential and proprietary information
// of Xilinx, Inc. and is protected under U.S. and
// international copyright and other intellectual property
// laws.
//
// DISCLAIMER
// This disclaimer is not a license and does not grant any
// rights to the materials distributed herewith. Except as
// otherwise provided in a valid license issued to you by
// Xilinx, and to the maximum extent permitted by applicable
// law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
// WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
// AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
// BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
// INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
// (2) Xilinx shall not be liable (whether in contract or tort,
// including negligence, or under any other theory of
// liability) for any loss or damage of any kind or nature
// related to, arising under or in connection with these
// materials, including for any direct, or any indirect,
// special, incidental, or consequential loss or damage
// (including loss of data, profits, goodwill, or any type of
// loss or damage suffered as a result of any action brought
// by a third party) even if such damage or loss was
// reasonably foreseeable or Xilinx had been advised of the
// possibility of the same.
//
// CRITICAL APPLICATIONS
// Xilinx products are not designed or intended to be fail-
// safe, or for use in any application requiring fail-safe
// performance, such as life-support or safety devices or
// systems, Class III medical devices, nuclear facilities,
// applications related to the deployment of airbags, or any
// other applications that could lead to death, personal
// injury, or severe property or environmental damage
// (individually and collectively, "Critical
// Applications"). Customer assumes the sole risk and
// liability of any use of Xilinx products in Critical
// Applications, subject only to applicable laws and
// regulations governing limitations on product liability.
//
// THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
// PART OF THIS FILE AT ALL TIMES.
******************************************************************************/
//   ____  ____
//  /   /\/   /
// /___/  \  /    Vendor             : Xilinx
// \   \   \/     Version            : 1.0
//  \   \         Application        : MIG
//  /   /         Filename           : ddr4_v2_2_1_ddr4_traffic_generator.v
// /___/   /\     Date Last Modified : $Date: 2014/09/03 $ 
// \   \  /  \    Date Created       : Mon May 20 2013
//  \___\/\___\
//
// Device           : UltraScale
// Design Name      : DDR4 SDRAM BEHAVIORAL EXAMPLE TB
// Purpose          : This is an  example behavioral test bench that drives
//                    the User Interface of DDR4 Memory Controller(MC).
//                    This example works for DDR4 memory controller generated from MIG.
//                    The intent of this behavioral test bench is for the MIG 
//                    users to get an estimate on the efficiency for a given 
//                    traffic pattern with the MIG controller. 
//  Description     : The test bench will pass the user supplied commands and
//                    address to the memory controller and measure the efficiency 
//                    for the given pattern. The efficiency is measured by the 
//                    occupancy of the DQ bus. The primary use of the test bench 
//                    is for efficiency measurements so no data integrity checks 
//                    are performed. Static data will be written into the memory 
//                    during write transactions and the same data will always be 
//                    read back. 
//
//                    ***********************************************************
//                    |                                                         |
//                    | Command_Repeat[7:0]    Address[35:0]     Command[3:0]   |
//                    |                                                         |
//                    ***********************************************************
//
//                    ==========================================================
//                    |                      Command[3:0]                       |
//                    ----------------------------------------------------------
//
//                       COMMAND     CODE           DESCRIPTION
//
//                        WRITE       0      This corresponds to the Write
//                                           operation that needs to be performed.
//
//                        READ        1      This corresponds to the Read
//                                           operation that needs to be performed.
//
//                        NOP         7      This corresponds to the idle 
//                                           situation for the bus.
//                     ==========================================================
//
//                     ==========================================================
//                     |                     Address[35:0]                      |
//                     ----------------------------------------------------------
//                     |   Rank[3:0]  |  Bank[3:0]  | Row[15:0] |  Column[11:0] |
//                     ----------------------------------------------------------
//
//                     All the address fields need to be entered in the hexadecimal
//                     format. All the address fields will be of width that is 
//                     divisible by four to enter in the hexadecimal format. 
//                     The test bench will only send the required bits of an address
//                     field based on the respective width parameter value set for 
//                     Rank, Bank, Row, Column to the memory controller.
//
//                     The address will be assembled based on the top level 
//                     "MEM_ADDR_ORDER" parameter and sent to the user interface(UI). 
//                     =============================================================
//
//                     Format of command repetition count:
//                     =============================================================
//                     |              Command_Repeat[7:0]                          |
//                     -------------------------------------------------------------
//
//                     The command repetition count is the number of time the 
//                     respective command is repeated at the user interface. 
//                     The address for each repetition is incremented by "8".
//                     The maximum repetition count will be 128. The test bench does
//                     not check for the column boundary and will wrap around if the 
//                     max column limit is reached during the increments.  128 commands
//                     will fill up the page. For any column address other than "0" the
//                     repetition count of 128 will end up crossing the column boundary
//                     and wrapping around to the start of the column address.
//
////////////////////////////////// End Description   ////////////////////////////////////////
//
//
//  Examples      :    =================================================================
//                     1. Single Read Pattern : 
//                        00_0_2_000F_00A_1 - This pattern is a single read from 
//                                            "10th column", "15th row" and "2nd bank".
//
//                     2. Single Write Pattern :
//                        00_0_1_0040_010_0 - This pattern is a single write to the
//                                            "32nd column","128th row" and "1st bank".
//
//                     3.	Single Write and Read to same address : 
//                        00_0_2_000F_00A_0 - This pattern is a single write to the 
//                                            "10th column","15th row" and "2nd bank".
//                        00_0_2_000F_00A_1 - This pattern is a single read from "
//                                            10th column","15th row" and "2nd bank".
//
//                     4. Multiple Writes and Reads with same address :
//                        0A_0_0_0010_000_0 - This corresponds to 11 writes with address
//                                            starting from "0" to "80"(command_repeat is 10)
//                        0A_0_0_0010_000_1 - This corresponds to 11 reads with address 
//                                            starting from "0" to "80"(command_repeat is 10)
//
//                     5.	Page Wrap during Writes :
//                        0A_0_2_000F_3F8_0 - This corresponds to eleven writes with column
//                                            address been wrapped to the starting of the 
//                                            page after one write.
//
/////////////////////////////////////    End Examples  //////////////////////////////////////////
/// Reference       :
// Revision History :
//*****************************************************************************

`timescale 1ps / 1ps

module ddr4_v2_2_1_ddr4_traffic_generator #(
  parameter APP_DATA_WIDTH   = 32,        // MC UI data bus width.
  parameter CMD_DEPTH        = 45,        // Number of commands driven to UI
  parameter COL_WIDTH        = 10,        // Memory Column Address bits
  parameter ROW_WIDTH        = 14,        // Memory Row Address bits
  parameter BANK_WIDTH       =  3,        // Memory Bank Address bits
  parameter BANK_GROUP_WIDTH =  2,        // Memory Bank Group Address
  parameter LR_WIDTH	     =  1,  	
  parameter RANK_WIDTH       =  1,        // Memory Rank Address bits
  parameter tCK              = 1250,      // Memory clock period in ps
  parameter MEM_ADDR_ORDER   = "BANK_ROW_COLUMN", // Address pattern order 
  parameter MEMORY_WIDTH     = 8,       // Memory_width 
  parameter ADDR_WIDTH       = 28,         // Address bus width of the MC UI 
  parameter S_HEIGHT 	     = 1
  )
  (
  // ********* ALL SIGNALS AT THIS INTERFACE ARE ACTIVE HIGH SIGNALS ********/
  input clk,                                  // UI clock
  input rst,                                  // UI reset signal.
  input init_calib_complete,                  // calibration done signal 
                                              // from UI.
  input app_rdy,                              // cmd fifo ready signal 
                                              // from UI.
  input app_wdf_rdy,                          // write data fifo ready 
                                              // signal from UI.
  input app_rd_data_valid,                    // read data valid signal from UI
  input [APP_DATA_WIDTH-1 : 0]  app_rd_data,  // read data from UI
  output [2 : 0]                app_cmd,      // command bus to the UI
  output [ADDR_WIDTH-1 : 0]     app_addr,     // address bus to the UI
  output                        app_en,       // command enable to UI
  output [(APP_DATA_WIDTH/8)-1 : 0] app_wdf_mask, // write data mask signal is 
                                              // tied to 0 in this TB.
  output [APP_DATA_WIDTH-1: 0]  app_wdf_data, // write data bus to the UI.
  output                        app_wdf_end,  // write burst end to the UI
  output                        app_wdf_wren, // write enable to the UI
  output                        cmp_error     // Tied to 0 in this TB as data 
                                              // integrity check is not done.
  );

//*****************************************************************************
// Fixed constant local parameters. DO NOT CHANGE these values. 
//*****************************************************************************
// Instruction encoding 
//************************
localparam READ  = 4'b0001;      // READ command value encoded to "1".
localparam WRITE = 4'b0000;      // WRITE command value encoded to "0".
localparam NOP   = 4'b0111;      // NOP command encode to "7".

// INTERNAL SIGNALS
reg  [51:0]                 mem[];                // Memory to load the lines 
                                                  // from stimulus.txt
reg  [3 :0]                 cmd              = 3; // Command instruction 
reg  [ADDR_WIDTH-1:0]       cmd_addr         = 0; // Command address
reg [7:0]                   ref_rpt_cnt      = 0; // reference Command repeat 
                                                  // count
reg [7:0]                   rpt_cnt          = 0; // To parse till reference 
                                                  // repeat count
reg                         cmd_en           = 0; // Command enable 
reg                         en_repetition    = 0; // To enable repetition
reg                         wr_trans         = 0; // enable for Write 
reg                         rd_trans         = 0; // enable for Read 
reg                         wr_cmd_complete  = 0; // Asserted when write 
                                                  // command completed
reg  [APP_DATA_WIDTH-1: 0]  wr_data          = 0; // Write data internal signal
reg                         wr_en            = 0; // Write enable signal
reg                         txn_done         = 0; // Asserted when any 
                                                  // transaction is done 
reg                         wr_rd_complete   = 0; // Asserted when all the 
                                                  // commands are served
reg  [COL_WIDTH-1:0]        col              = 0; // column Address
reg  [ROW_WIDTH-1:0]        row              = 0; // Row Address
reg  [BANK_WIDTH-1:0]       bank             = 0; // Bank Address
reg  [BANK_GROUP_WIDTH-1:0] bank_group       = 0; // Bank Group Address
reg  [LR_WIDTH-1:0]	    lr		     = 0; // Logical address	
reg  [RANK_WIDTH-1:0]       rank             = 0; // Rank Address

integer                     wr_cmd_cnt       = 0; // Write command count 
integer                     rd_cmd_cnt       = 0; // Read command count
integer                     sz               = 0; // Depth of memory loaded 
                                                  // with stimulus.txt  
integer                     stim_depth       = 0; // To store the calculated 
                                                  // stimulus file depth

assign cmp_error=0;                                           

//*****************************************************************************
// Command enable signal generation to the UI 
//*****************************************************************************
// The app_en signal is used to qualify the command to the UI. cmd_en is 
// asserted for valid read or write transaction.
// The command is considered accepted by the UI When both app_rdy and cmd_en 
// signals are asserted in the same clock cycle.
//*****************************************************************************
always @ (*) begin  
  cmd_en = (((wr_trans & app_wdf_rdy) | rd_trans) & app_rdy) ;
  wr_en  = (wr_trans & app_wdf_rdy & app_rdy);
end

assign app_en = cmd_en & (app_rdy) ;

//*****************************************************************************
// Command and Address generation to the UI
//*****************************************************************************
// The app_cmd signal is the command issued to the UI.
// app_addr is the address for the request currently being submitted to the UI.
//*****************************************************************************
assign app_cmd  = cmd;
    
assign app_addr = cmd_addr;

//*****************************************************************************
// Write enable signal generation to the UI
//*****************************************************************************
// The app_wdf_wren signal indicates that the data on the app_wdf_data 
// bus is valid.
// When the app_wdf_wren is high, the write data is written into the 
// UI write data fifo.
//*****************************************************************************
assign app_wdf_wren = wr_en & (app_rdy) ;


//*****************************************************************************
// Write end signal generation to the UI
//*****************************************************************************
// The app_wdf_end signal is the write end information to the UI, 
//*****************************************************************************
assign app_wdf_end = wr_en & (app_rdy)  ;

//*****************************************************************************
// Write data generation to the UI 
//*****************************************************************************
// The app_wdf_data bus is the write data issued to the UI.
// For 4:1 clock ratio in BL8 Mode, the data has to be provided for the entire
// BL8 burst in one clock cycle (User interface clock cycle).
//
// The data has to be provided in the following format:
// FALL3->RISE3->FALL2->RISE2->FALL1->RISE1->FALL0->RISE0
// 
// For an 16 bit interface, 16 * 8 = 128 bits of data will be provided in the
// each clock cycle. LSB 16-bits corresponds to RISE0 and MSB 16-bits 
// corresponds to FALL3.
//*****************************************************************************
assign app_wdf_data = {(APP_DATA_WIDTH/ADDR_WIDTH){wr_data}};

//*****************************************************************************
// Write data mask to the MC
// ** The write data mask is set to zero in this module **
//*****************************************************************************
// The app_wdf_mask signal is not used so it is tied to 0 in this example.
// If the mask signal need to be toggled, the timing is same as write data.
//*****************************************************************************
assign app_wdf_mask = 0 ;

//*****************************************************************************
// Loading the command from the stimulus file
//*****************************************************************************
// when reset,cmd is set to NOP
// when not in reset,cmd is loaded with current parsed stimulus file line
// resembled by sz where wr_trans and rd_trans are set to zero.
//*****************************************************************************
always @(posedge clk) 
begin
  if(rst) begin
    cmd      = 7; 
    txn_done = 1;
    wr_trans = 1'b0;
    rd_trans = 1'b0;
  end else if (init_calib_complete && txn_done && app_rdy && (sz<stim_depth) )
  begin
    cmd           = mem[sz][3:0]; 
    txn_done      = 0;
//    wr_trans      = 1'b0;
//    rd_trans      = 1'b0;
    en_repetition = 1;
    rpt_cnt       = 0;
  end
end

//*****************************************************************************
// Loading the Address and repetition count from current line read from 
// the stimulus file 
//*****************************************************************************
// This block loads the fields of address respectively col,row,bank,rank and 
// concatenate it to form the respective address to be sent based on 
// MEM_ADDR_ORDER parameter.
// 
// It also loads the repetition count for that respective command and address
// read from the current line which increments with "8" to parse through
// the repetition count.
//*****************************************************************************
always @(posedge clk)
begin
  // Write command processing
  if(cmd == WRITE && app_rdy && app_wdf_rdy && (en_repetition == 1)) begin
#0ps;
     wr_trans    = 1'b1;
     rd_trans    = 1'b0;
     col         = mem[sz][15:4];
     row         = mem[sz][31:16];
     rank        = mem[sz][39:36];
     bank        = mem[sz][33:32];
     bank_group  = mem[sz][35:34];
    if(S_HEIGHT >=  2) begin	    	  
     lr          = mem[sz][43:40];
     ref_rpt_cnt = mem[sz][51:44];
	  end else   
     ref_rpt_cnt = mem[sz][47:40];
     wr_data     =  $random;
     wr_cmd_cnt  = wr_cmd_cnt + 1;
      if (MEM_ADDR_ORDER == "ROW_BANK_COLUMN") begin
	if(S_HEIGHT >=  2)	    	  
          cmd_addr = {rank,lr,row,bank_group,bank,col};
	else   
          cmd_addr = {rank,row,bank_group,bank,col};
      end else if(MEM_ADDR_ORDER == "BANK_ROW_COLUMN") begin
        if(S_HEIGHT >=  2)
          cmd_addr = {rank,lr,bank_group,bank,row,col};
	else  
          cmd_addr = {rank,bank_group,bank,row,col};
      end else if(MEM_ADDR_ORDER == "ROW_COLUMN_LRANK_BANK") begin
          cmd_addr = {rank,row,col[COL_WIDTH-1:3],lr,bank,bank_group,col[2:0]};
      end else if(MEM_ADDR_ORDER == "ROW_LRANK_COLUMN_BANK") begin
          cmd_addr = {rank,row,lr,col[COL_WIDTH-1:3],bank,bank_group,col[2:0]};
      end else begin
        if(S_HEIGHT >=  2)
          cmd_addr = {rank,lr,row,col[COL_WIDTH-1:3],bank,bank_group,
                      col[2:0]};
	else		      
          cmd_addr = {rank,row,col[COL_WIDTH-1:3],bank,bank_group,
                      col[2:0]};
      end
 
     if(ref_rpt_cnt>0) 
        en_repetition = 1;
     else 
        en_repetition = 0;
  end 
  // Read command processing
  else if(cmd == READ && app_rdy &&  (en_repetition == 1))begin 
     rd_trans    = 1'b1;
     wr_trans    = 1'b0;
     col         = mem[sz][15:4];
     row         = mem[sz][31:16];
     rank        = mem[sz][39:36];
     bank        = mem[sz][33:32];
     bank_group  = mem[sz][35:34];
    if(S_HEIGHT >=  2) begin	    	  
     lr          = mem[sz][43:40];
     ref_rpt_cnt = mem[sz][51:44];
	  end else   
     ref_rpt_cnt = mem[sz][47:40];
     rd_cmd_cnt  = rd_cmd_cnt + 1;
      if (MEM_ADDR_ORDER == "ROW_BANK_COLUMN") begin
	if(S_HEIGHT >=  2)	      
          cmd_addr = {rank,lr,row,bank_group,bank,col};
	else  
          cmd_addr = {rank,row,bank_group,bank,col};
      end else if(MEM_ADDR_ORDER == "BANK_ROW_COLUMN") begin
        if(S_HEIGHT >=  2)
          cmd_addr = {rank,lr,bank_group,bank,row,col};
	else  
          cmd_addr = {rank,bank_group,bank,row,col};
      end else if(MEM_ADDR_ORDER == "ROW_COLUMN_LRANK_BANK") begin
          cmd_addr = {rank,row,col[COL_WIDTH-1:3],lr,bank,bank_group,col[2:0]};
      end else if(MEM_ADDR_ORDER == "ROW_LRANK_COLUMN_BANK") begin
          cmd_addr = {rank,row,lr,col[COL_WIDTH-1:3],bank,bank_group,col[2:0]};
      end else begin
        if(S_HEIGHT >=  2) 
          cmd_addr = {rank,lr,row,col[COL_WIDTH-1:3],bank,bank_group,
                      col[2:0]};
	else		      
          cmd_addr = {rank,row,col[COL_WIDTH-1:3],bank,bank_group,
                      col[2:0]};
      end
     if(ref_rpt_cnt>0) 
        en_repetition = 1;
     else 
        en_repetition = 0;
   end 
   // NOP command processing where address field is ignored
   else if(cmd == NOP && app_rdy && (en_repetition == 1)) begin
     ref_rpt_cnt =  mem[sz][47:40];
     rd_trans    = 1'b0;
     wr_trans    = 1'b0;
     if(ref_rpt_cnt>0) 
       en_repetition = 1;
     else 
       en_repetition = 0;
   end
end

//*****************************************************************************
// Parsing the repetition count from the stimulus file for a respective command
//*****************************************************************************
// This block parses through the repetition count loaded while reading a
// particular stimulus line.
// Wait for the current command to execute and if repetition count is non-zero, 
// increment its column address by "8" for every count honored.
//*****************************************************************************
always @(posedge clk)
begin
 // Checking for current command to execute and also parsed repetition 
 // count to honor until it meets reference repetition count.
 if ( (rpt_cnt) < ref_rpt_cnt && !en_repetition) begin
       txn_done = 0;
       // Honoring repeat count for write command.
       if(cmd == WRITE) begin
         if( app_rdy && app_wdf_rdy) begin
           col                     = col+8;
           cmd_addr[COL_WIDTH-1:0] = col;
           wr_data                 =  $random;
           rpt_cnt                 =  rpt_cnt+1;
           wr_cmd_cnt              = wr_cmd_cnt + 1;
           if (MEM_ADDR_ORDER == "ROW_BANK_COLUMN") begin
	      if(S_HEIGHT >=  2)					   
               cmd_addr = {rank,lr,row,bank_group,bank,col};
	      else  
               cmd_addr = {rank,row,bank_group,bank,col};
           end else if(MEM_ADDR_ORDER == "BANK_ROW_COLUMN") begin
	      if(S_HEIGHT >=  2)	
               cmd_addr = {rank,lr,bank_group,bank,row,col};
	      else 
               cmd_addr = {rank,bank_group,bank,row,col};
      	   end else if(MEM_ADDR_ORDER == "ROW_COLUMN_LRANK_BANK") begin
	          cmd_addr = {rank,row,col[COL_WIDTH-1:3],lr,bank,bank_group,col[2:0]};
	   end else if(MEM_ADDR_ORDER == "ROW_LRANK_COLUMN_BANK") begin
	          cmd_addr = {rank,row,lr,col[COL_WIDTH-1:3],bank,bank_group,col[2:0]};
           end else begin
	      if(S_HEIGHT >=  2)	
               cmd_addr = {rank,lr,row,col[COL_WIDTH-1:3],bank,bank_group,
                           col[2:0]};
	      else 					   
               cmd_addr = {rank,row,col[COL_WIDTH-1:3],bank,bank_group,
                           col[2:0]};
           end
         end
       end 
       // Honoring repeat count for Read command.
       else if(cmd == READ )begin
         if(app_rdy) begin
           col                     = col+8;
           cmd_addr[COL_WIDTH-1:0] = col;
           rpt_cnt                 = rpt_cnt+1;
           rd_cmd_cnt              = rd_cmd_cnt + 1;
           if (MEM_ADDR_ORDER == "ROW_BANK_COLUMN") begin
	      if(S_HEIGHT >=  2)			   
	       cmd_addr = {rank,lr,row,bank_group,bank,col};
	      else 			       
	       cmd_addr = {rank,row,bank_group,bank,col};
           end else if(MEM_ADDR_ORDER == "BANK_ROW_COLUMN") begin
	       if(S_HEIGHT >=  2)
	        cmd_addr = {rank,lr,bank_group,bank,row,col};
	       else 		
	        cmd_addr = {rank,bank_group,bank,row,col};
      	   end else if(MEM_ADDR_ORDER == "ROW_COLUMN_LRANK_BANK") begin
	          cmd_addr = {rank,row,col[COL_WIDTH-1:3],lr,bank,bank_group,col[2:0]};
	   end else if(MEM_ADDR_ORDER == "ROW_LRANK_COLUMN_BANK") begin
	          cmd_addr = {rank,row,lr,col[COL_WIDTH-1:3],bank,bank_group,col[2:0]};
           end else begin
	      if(S_HEIGHT >=  2)
               cmd_addr = {rank,lr,row,col[COL_WIDTH-1:3],bank,bank_group,
                           col[2:0]};
	      else				   
               cmd_addr = {rank,row,col[COL_WIDTH-1:3],bank,bank_group,
                           col[2:0]};
           end
         end
       end
       // Honoring repeat count in case of NOP.
       else begin 
          rpt_cnt = rpt_cnt+1;
       end
  end
  // Checking parsed repeat count against loaded reference repeat count to 
  // load next stimulus line.
  else if((ref_rpt_cnt == rpt_cnt) && app_rdy && (sz<(stim_depth))) begin
    if(app_wdf_rdy)
	    sz       = sz+1;
    else if(!app_wdf_rdy) 
	    sz 	     = sz-1;
#0ps;
    txn_done = 1;
  end
  // Wait to send the current command when a stimulus line is read and repeat 
  // count is non-zero.
  else begin
    en_repetition = 0;
  end
end

//*****************************************************************************
// Checking the parsed stimulus lines against actual number of stimulus lines
//*****************************************************************************
// This block checks for parsed stimulus lines against actual stimulus lines
// If parsed lines are equal to number stimulus lines asserts all transaction 
// to complete
//*****************************************************************************
always@(posedge clk) begin
  if( (sz == stim_depth)) begin
    wr_rd_complete = 1;
    cmd_en         = 0;
  end
end
  
integer file;
integer numerator,denominator;
real percentage_bus_util;
time cal_done,end_of_stimulus;

//*****************************************************************************
// Calculating the number of lines in stimulus file to parse through
//*****************************************************************************
task cal_stim_depth();
 while(!$feof (file))begin
  string line;
  if($fgets(line,file)) begin
    stim_depth++;
    end
  end
  $display("no_of_lines = %0d",stim_depth);
endtask

//*****************************************************************************
// Calculating the percentage of bus utilization
//*****************************************************************************
//  Opening the stimulus file "ddr4_v2_2_1_ddr4_stimulus.txt" for reading.
//  Calculate the depth of stimulus file.
//  Read the stimulus file contents into memory.
//  Small delay added to wait till last command gets placed onto controller
//  after all transactions are driven(not accurate).
//  Open a file "ddr4_v2_2_1_ddr4_band_width_cal.txt" to write result into it.
//  The bus utilization is calculated at the user interface taking 
//  total number of Reads and Writes into consideration.
//  The following equation is used:
//                  ((rd_command_cnt + wr_command_cnt) ×(BURST_LEN / 2) ×100) 
//  bw_cumulative = -----------------------------------------------------------
//                  ((end_of_stimulus - calib_done) / tCK);
//
//            1. BL8 takes four memory clock cycles.
//            2. end_of_stimulus is the time when all the commands are done.
//            3. calib_done is the time when the calibration is done.
//*****************************************************************************
initial
begin 
if(S_HEIGHT ==  2) begin	    	  
  file=$fopen("ddr4_v2_2_1_ddr4_stimulus_mem_x4_x8_3ds_2h.txt","r");
end else if(S_HEIGHT == 4) begin 
  file=$fopen("ddr4_v2_2_1_ddr4_stimulus_mem_x4_x8_3ds_4h.txt","r");
end else begin 
  if(MEMORY_WIDTH == 16)
  file=$fopen("ddr4_v2_2_1_ddr4_stimulus_mem_x16.txt","r");
  else
  file=$fopen("ddr4_v2_2_1_ddr4_stimulus_mem_x4_x8.txt","r");
end
  cal_stim_depth();
  mem=new[stim_depth];
if(S_HEIGHT ==  2) begin	    	  
  $readmemh("ddr4_v2_2_1_ddr4_stimulus_mem_x4_x8_3ds_2h.txt",mem);
end if(S_HEIGHT == 4) begin 
  $readmemh("ddr4_v2_2_1_ddr4_stimulus_mem_x4_x8_3ds_4h.txt",mem);
end else begin 
  if(MEMORY_WIDTH == 16)
  $readmemh("ddr4_v2_2_1_ddr4_stimulus_mem_x16.txt",mem);
  else
  $readmemh("ddr4_v2_2_1_ddr4_stimulus_mem_x4_x8.txt",mem);
end
  wait(init_calib_complete==1);
  cal_done = $time;
  wait (wr_rd_complete==1);
  #10;
  end_of_stimulus     = $time;
  file                = $fopen("ddr4_v2_2_1_ddr4_band_width_cal.txt");
  numerator           = ((wr_cmd_cnt+rd_cmd_cnt)*400);
  denominator         = (end_of_stimulus-cal_done)/tCK;
  percentage_bus_util = (((numerator)/(denominator)));
  $fwrite(file,"  Number of Writes = %0d\n",wr_cmd_cnt);
  $fwrite(file,"  Number of Reads  = %0d\n",rd_cmd_cnt);
  $fwrite(file,"  Calibration is done at %0t\n",cal_done);
  $fwrite(file,"  Transactions ended at %0t\n",end_of_stimulus);
  $fwrite(file,"  Percentage of Bus utilization =  %f\n",percentage_bus_util);
  $display ("=================================================================");
  $display ("Performance Simulation Completed Successfully!!!");
  $display ("Data integrity check is not performed in performance simulation");
  $display ("=================================================================");
  $display ("");
  $display ("*****************************************************************");
  $display ("                     PERFORMANCE STATISTICS ");
  $display ("*****************************************************************");
  $display ("\tNumber of Writes                : %0d",wr_cmd_cnt);
  $display ("\tNumber of Reads                 : %0d",rd_cmd_cnt);
  $display ("\tCalibration is done at          : %0t",cal_done);
  $display ("\tTransactions ended at           : %0t",end_of_stimulus);
  $display ("\tPercentage of Bus utilization   : %f",percentage_bus_util);
  $display ("\n");
  $finish;
end

endmodule


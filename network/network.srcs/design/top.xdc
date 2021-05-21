##################################################################################
#  Company:     A*STAR IHPC
#  Engineer:    Gerlinghoff Daniel
#  Create Date: 18/05/2021
#
#  Description: Pin assignments and I/O contraints
#
##################################################################################

set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]

# Clock signal
set_property PACKAGE_PIN H19 [get_ports "clk_user_p"]
set_property IOSTANDARD DIFF_HSTL_I_18 [get_ports "clk_user_p"]

set_property PACKAGE_PIN H18 [get_ports "clk_user_n"]
set_property IOSTANDARD DIFF_HSTL_I_18 [get_ports "clk_user_n"]

create_clock -period 10 [get_ports "clk_user_p"]

set_property PACKAGE_PIN BA34 [ get_ports "sys_clk_p" ]
set_property IOSTANDARD DIFF_HSTL_I_12 [ get_ports "sys_clk_p" ]

set_property PACKAGE_PIN BB34 [ get_ports "sys_clk_n" ]
set_property IOSTANDARD DIFF_HSTL_I_12 [ get_ports "sys_clk_n" ]

create_clock -period 5 [get_ports "sys_clk_p"]

# UART signals
set_property PACKAGE_PIN BC28 [get_ports "uart_rst_n"]
set_property IOSTANDARD LVCMOS18 [get_ports "uart_rst_n"]
set_property PACKAGE_PIN BE28 [get_ports "uart_txd"]
set_property IOSTANDARD LVCMOS18 [get_ports "uart_txd"]
set_property PACKAGE_PIN BF29 [get_ports "uart_rts"]
set_property IOSTANDARD LVCMOS18 [get_ports "uart_rts"]
set_property PACKAGE_PIN BD28 [get_ports "uart_rxd"]
set_property IOSTANDARD LVCMOS18 [get_ports "uart_rxd"]
set_property PULLUP TRUE [get_ports "uart_rxd"]
set_property PACKAGE_PIN BF28 [get_ports "uart_cts"]
set_property IOSTANDARD LVCMOS18 [get_ports "uart_cts"]


set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design] 

# Push buttons
set_property PACKAGE_PIN AA33 [ get_ports "sys_rst" ]
set_property IOSTANDARD LVCMOS12 [ get_ports "sys_rst" ]
set_false_path -through [ get_ports "sys_rst" ]
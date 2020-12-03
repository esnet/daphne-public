Project under development


Overview
---------

Testing Network Performance before DTN-DTN transfer (designed for big data transfers)
------------------------------------------------------------------------------

To Add Abstract

Version 1)
Network Measurement Tools used - Iperf, Traceroute.

- iperf3 -c <server_ip>

- traceroute <server_ip>

Run the following on your DTN terminal:

- bash <script_name (.sh)> <no._of_runs> <server_ip> <file_transfer_size> <file_output>

e.g bash preflightcheck.sh 5 192.5.87.205 1G testresult.txt

Version 2)
Without Iperf

- python3 <scriptname> -H <TargetHostIPaddress> -F <targetFile> -I <no. of iterations>   

 e.g python3 netpreflight_ssh_traceroute.py -H 192.5.87.127 -F d-icon.png -I 5

on HOST_B: No action is required on host_B     
Specify the TargetHost IP address for the traceroute command     
                                                                         


Authors
---------
- Bashir Mohammed
- Mariam Kiran
- Divneet Kaur

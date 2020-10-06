#!/bin/bash

if ! [ -x "$(type -P iperf3)" ]; then
  echo "ERROR: script requires iperf3"
  echo "For Debian and friends get it with 'apt-get install iperf3'"
  echo "If you have it, perhaps you don't have permissions to run it, try 'sudo $(basename $0)'"
  exit 1
fi

if [ "$#" -ne "4" ]; then
  echo "ERROR: script needs five arguments, where:"
  echo
  echo "1. Number of times to repeat test (e.g. 10)"
  echo "2. Host running 'iperf3 -s' (e.g. somehost)"
  echo "3. Number of file size (e.g 1G)"
  echo "4. Output file for results (e.g report.txt)"
  echo
  echo "Example:"
  echo "  $(basename $0) 10 somehost 1G report.txt"
  echo
  echo "The above will run 'iperf3 -c' 10 times on the client and report totals and average and write results to a file called report.txt"
  exit 1
else
  runs=$1
  host=$2
  size=$3
  log=$4
fi



if [ -f $log ]; then
  echo removing $log
  rm $log
fi

echo "=================================================================="
echo " Results"
echo "=================================================================="
echo " target host .... $host"
echo "------------------------------------------------------------------"
echo " taking measurements......."
echo "------------------------------------------------------------------"

for run in $(seq 1 $runs); do
  echo "Iperf3 Results ... \n" >> log.txt
  echo "==================================================================" >> log.txt
  iperf -c $host -f m -n $size  >> $log
  echo "Traceroute Results ... \n" >> log.txt
  echo "==================================================================" >> log.txt
  traceroute $host >> $log
  #echo "Iperf3 file transfer Results ... \n" >> log.txt
  #echo "==================================================================" >> log.txt
  #iperf -c $host -n $size >> $log
  #echo "netperf Results ... \n" >> log.txt
  #echo "==================================================================" >> log.txt
  #netperf -H $host >>log
  echo -e " run $run: \t $(awk '/Bandwidth/ {getline}; END{print $7, $8}' $log)"
done

avg=$(awk -v runs=$runs '/Bandwidth/ {getline; sum+=$7; avg=sum/runs} END {print avg}' $log)


echo "------------------------------------------------------------------"
echo " average ....... $avg Mbits/sec"
echo
echo "see $log for details"
##########################################################################################
#                               Netpreflight Usage:                                      #
##########################################################################################
#                                                                                        #
# on HOST_A: On your terminal run the command below with the following arguements        # 
#                                                                                        #
# #python3 <scriptname> -H <TargetHostIPaddress> -F <targetFile> -I <no. of iterations>  #
#                                                                                        #
# e.g python3 netpreflight_ssh_traceroute.py -H 192.5.87.127 -F d-icon.png -I 5          #
#                                                                                        #
# on HOST_B: No action is required on host_B                                             #
#                                                                                        #
# Specify the TargetHost IP address for the traceroute command                           #
#                                                                                        #
##########################################################################################
import socket, os, sys, optparse, time
import subprocess

#output file-results
outfile = 'results.csv'
BufferSize = 1024

def retBanner(ip, port=22):
    sock = socket.socket()
    sock.connect((ip, port))
    sock.send(b'Gabbage')
    banner = sock.recv(1024)
    return banner


def download_file(targetHost, targetFile):
    connection_test = retBanner(targetHost)
    if connection_test is None:
        print('Port is closed')
        exit(0)
    else:
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.connect((targetHost, 22))
        connection.send(bytes(targetFile, encoding='utf-8'))
        with open('./{}'.format(targetFile.split('/')[-1]), 'wb') as ff:
            while True:
                data = connection.recv(BufferSize)
                if not data:
                    break
                ff.write(data)
            ff.close()
        connection.close()
        print('Data Downloaded Succesfully..!')

def main():
    parser = optparse.OptionParser('usage %prog -H <targetHost> -F <targetFile> -I <iterations>')
    parser.add_option('-H', dest='targetHost', type='string', help='Specify a connection port')
    parser.add_option('-F', dest='targetFile', type='string', help='Specify File omn remote to download')
    parser.add_option('-I', dest='iterations', type='int', help='Specify number of iterations')

    (options, args) = parser.parse_args()
    host = options.targetHost
    file = options.targetFile
    iterations = options.iterations

    headings = ["Iter", "Throughput", "Time(s)", "Buffer"]
    data = []

    for iteration in range(iterations):
        start = time.time()
        download_file(targetHost=host, targetFile=file)
        end = time.time()

        lapse = round(((end - start)/1000), 3)
        with open('outfile', 'a') as ff:
            tp = round((BufferSize*0.001) / (lapse+0.000001), 3)
            smallist = [iteration, tp, lapse, BufferSize]
            data.append(smallist)
            ff.write('iteration {},{},{},{}\n'.format(iteration, tp, lapse, BufferSize))
        ff.close()
    format_row = "{:>12}" * (len(headings) + 1)
    print(format_row.format("", *headings))
    for row in data: 
        print(format_row.format('', *row))

    file_ = open('traceresult.txt', 'w+')
    subprocess.run('traceroute 67.205.158.239', shell=True, stdout=file_)
    file_.close()

    with open('traceresult.txt', 'r') as f:
            print('netpreflight',f.read())
       

if __name__ == '__main__':
    main()


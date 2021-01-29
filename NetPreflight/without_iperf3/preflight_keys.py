############################################################################################################################
#                                          Netpreflight Usage using Public Keys:                                           #
############################################################################################################################
#                                                                                                                          #
# on HOST_A: On your terminal run the command below with the following arguements                                          #
#                                                                                                                          #
# #python <scriptname> -H <TargetHostIPaddress> -K <KeyFilepath>  -F <targetFile> -I <no. of iterations>                   #
#                                                                                                                          #
# e.g python preflight_keys.py -H 67.205.158.239 -K <KeyFilepath> -F /root/largefiles/ -I 5                                #
# e.g python preflight_keys.py -H 138.68.10.107 -K <KeyFilepath> -F /root/largefiles/ -I 5                                 #                                                        #
# on HOST_B: No action is required on host_B                                                                               #
#                                                                                                                          #
# Specify the TargetHost IP address for the traceroute command                                                             #
# For Example:                                                                                                             #
#                                                                                                                          # 
# python preflight_keys.py -H 192.5.87.20 -K /Users/bashirm/Downloads/uc-mc4n-key.pem -F /home/cc/experiments/5MB.zip -I 5 #
# python preflight_keys.py -H 192.5.87.20 -K /home/cc/experiments/uc-mc4n-key.pem -F /home/cc/experiments/5MB.zip -I 5     #                                                                                        #
# Requirements: sudo pip install paramiko                                                                                  #
############################################################################################################################

import socket, os, sys, optparse, time
import subprocess
import paramiko

import getpass

ssh = paramiko.SSHClient()
outfile = 'results.csv'
BufferSize = 1024

username = ''
key = ''

def userprompt():
    username = input("Hello! Welcome to Netpreflight Tool! \n\nUsername: ") 
    key = getpass.getpass('Password :: ')
    print('key',key)


def retBanner(ip, port=22):
    sock = socket.socket()
    sock.connect((ip, port))
    sock.send(b'Gabbage')
    banner = sock.recv(BufferSize)
    return banner

def download_fileV3(targetHost, targetFile, username,key):
    k = paramiko.RSAKey.from_private_key_file(key)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print ("connecting")
    c.connect( hostname = targetHost, username = username, pkey = k )
    print('..................................')
    print('Starting to download file at {}...!'.format(targetFile))
    sftp = c.open_sftp()
    localpath = targetFile.split("/")[-1]
    remotepath = targetFile
    sftp.get(remotepath,localpath)
    sftp.close()
    c.close()

def main():
    parser = optparse.OptionParser('usage %prog -H <targetHost> -F <targetFile> -I <iterations>')
    parser.add_option('-H', dest='targetHost', type='string', help='')
    parser.add_option('-K', dest='key', type='string', help=' Specify the location of your key')
    parser.add_option('-F', dest='targetFile', type='string', help='Specify File on remote to download')
    parser.add_option('-I', dest='iterations', type='int', help='Specify number of iterations')

    (options, args) = parser.parse_args()
    host = options.targetHost
    file = options.targetFile
    key= options.key
    iterations = options.iterations

    username = input("Hello! Welcome to Netpreflight Tool! \n\nUsername: ") 

    headings = ["Iter", "Throughput", "Time(s)", "Buffer"]
    data = []

    for iteration in range(iterations):
        start = time.time()
        download_fileV3(targetHost=host, targetFile=file, username=username, key=key)
        end = time.time()

        lapse = round(((end - start)/10), 3)
        with open(outfile, 'a') as ff:
            tp = round(((BufferSize*8)) / (lapse+0.000001), 3)
            tp /= 100
            smallist = [iteration, tp, lapse, BufferSize]
            data.append(smallist)
            ff.write('iteration {},{},{},{}\n'.format(iteration, tp, lapse, BufferSize))
        ff.close()
    format_row = "{:>12}" * (len(headings) + 1)
    print(format_row.format("", *headings))
    for row in data: 
        print(format_row.format('', *row))

    file_ = open('traceresult.txt', 'w+')
    subprocess.run('traceroute '+ host, shell=True, stdout=file_)
    file_.close()

    with open('traceresult.txt', 'r') as f:
            print('netpreflight',f.read())
       

if __name__ == '__main__':
    main()



##########################################################################################
#                               Netpreflight Usage with Pass:                            #
##########################################################################################
#                                                                                        #
# on HOST_A: On your terminal run the command below with the following arguements        # 
#                                                                                        #
# #python <scriptname> -H <TargetHostIPaddress> -F <targetFile> -I <no. of iterations>   #
#                                                                                        #
# e.g python preflight_pass.py -H 67.205.158.239 -F /root/largefiles/100MB.zip -I 5      #
# e.g python preflight_pass.py -H 138.68.10.107 -F /root/largefiles/100MB.zip -I 5       #                                                        #
#                                                                                        #
# on HOST_B: No action is required on host_B                                             #
#                                                                                        #
# Remember to specify the TargetHost IP address for Example.                             #
# user: root                                                                             #
# pw: Password1Pass                                                                      #
# Requirements: sudo pip install paramiko                                                #
##########################################################################################
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

def download_fileV2(targetHost, targetFile, username,key):
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print('Trying to Connect to host at {}...!'.format(targetHost))
  

    ssh.connect(hostname=targetHost, username=username, password=key)
    print('Connected to host at {}...!'.format(targetHost))

    print('..................................')
    print('Starting to download file at {}...!'.format(targetFile))
    sftp = ssh.open_sftp()
    localpath = targetFile.split("/")[-1]
    remotepath = targetFile
    sftp.get(remotepath,localpath)
    sftp.close()
    ssh.close()


def download_file(targetHost, targetFile):
    connection_test = retBanner(targetHost)
    if connection_test is None:
        print('Port is closed')
        exit(0)
    else:
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Connected to host at {}...!'.format(targetHost))
        connection.connect((targetHost, 22))
        #print('Downloading..!')
        connection.send(bytes(targetFile, encoding='utf-8'))
        #print('Writing results..!')
        with open('./{}'.format(targetFile.split('/')[-1]), 'wb') as ff:
            while True:
                data = connection.recv(BufferSize)
                if not data:
                    break
                ff.write(data)
            ff.close()
        connection.close()
    #print('Data Downloaded Succesfully..!')

def main():
    parser = optparse.OptionParser('usage %prog -H <targetHost> -F <targetFile> -I <iterations>')
    parser.add_option('-H', dest='targetHost', type='string', help='Specify a connection port')
    parser.add_option('-F', dest='targetFile', type='string', help='Specify File omn remote to download')
    parser.add_option('-I', dest='iterations', type='int', help='Specify number of iterations')

    (options, args) = parser.parse_args()
    host = options.targetHost
    file = options.targetFile
    iterations = options.iterations

    username = input("Hello! Welcome to Netpreflight Tool! \n\nUsername: ") 
    key = getpass.getpass('Password :: ')
    #print('key',key)

    headings = ["Iter", "Throughput", "Time(s)", "Buffer"]
    data = []

    for iteration in range(iterations):
        start = time.time()
        download_fileV2(targetHost=host, targetFile=file, username=username, key=key)
        end = time.time()

        lapse = round(((end - start)/1000), 3)
        with open(outfile, 'a') as ff:
            tp = round(((BufferSize*8)) / (lapse+0.000001), 3)
            tp /= 1000000
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



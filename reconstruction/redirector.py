import serial
import socket
import sys
import time
from struct import unpack,pack

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.connect(('127.0.0.1', 1234))
#s.connect(('risa.local', 1234))
s.connect(('lente.local', 1234))

packetlen = 22
#packetlen = 36

if len(sys.argv) > 1:
    fname = sys.argv[1]
    f = open(fname, "r")
    starttime = time.time()
    firsttime = unpack('<d', f.read(8))[0]
    f.seek(0)
    while True:
        t = unpack('<d', f.read(8))[0]
        timediff = (t - firsttime) - (time.time() - starttime)
        if timediff > 0:
            time.sleep(timediff)
        s.sendall(f.read(packetlen))
    sys.exit(1)

# Establish a serial connection and clear the buffer
ser = serial.Serial("/dev/ttyACM0", 38400);
ser.flushInput()
while ser.inWaiting():
	ser.read(True)
beef = '\xef' + '\xbe'
buffer = ''

f = open("output.dat", "w")

# Keep on listening for multi-Spin packets
while True:
    buffer = buffer + ser.read(ser.inWaiting())
    if beef in buffer:
        lines = buffer.split(beef, 1)
        binaryPacket = lines[-2]
        buffer = lines[-1]
	if len(binaryPacket) != packetlen:
		continue
        #s.send('\xDE\xAD')
        s.sendall(binaryPacket)

        f.write(pack('<d', time.time()))
        f.write(binaryPacket)

        spinPacket = unpack('<Hb' + (len(binaryPacket) - 4) * 'b' + 'b', binaryPacket)
        #print(spinPacket)
    else:
        time.sleep(0.001)

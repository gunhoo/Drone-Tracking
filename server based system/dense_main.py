import glob
import sys
import pyaudio
import wave
import numpy as np
import tensorflow as tf
import librosa
from socket import *
from header import *

if len(sys.argv) < 4:
    print("Compile error : python main.py [nodeNum] [posX] [posY]")
    exit(1)

FORMAT = pyaudio.paInt16
NODE = sys.argv[1]
posX = sys.argv[2]
posY = sys.argv[3]

# connection
clientSocket = socket(AF_INET, SOCK_STREAM)
try:
    clientSocket.connect(('192.168.123.3',21535))
except Exception as e:
    print('cannot connect to the server;', e)
    exit()

#send node info
nodeInfo = NODE + ":" + posX + ":" + posY
clientSocket.send(nodeInfo.encode())

# open pyaudio
p = pyaudio.PyAudio()
stream = p.open(format = FORMAT,
                channels = CHANNELS,
                rate = RATE,
                input = True,frames_per_buffer = CHUNK,
                #input_device_index = 0,
                #output_device_index = 0)
                )

# start loop
print("Start recording...")
while True:
    try:
        # initailize values
        printer("Start")
        sess = tf.Session()
        init = tf.global_variables_initializer()
        tf.reset_default_graph()
        sess.run(init)
        frames = []

        # recording
        for i in range(0, int(RATE/CHUNK*RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        printer("Record")
        # record/laod wav files
        file_saver(frames, wave, p)
        files = glob.glob(path)
        raw_data = load(files)
        printer("I/O")

        ### send packet
        clientSocket.send(raw_data)
        printer("TCP")
    # exception handle
    except KeyboardInterrupt:
        print("wait seconds to terminate...")
        stream.stop_stream()
        stream.close()
        p.terminate()
        clientSocket.close()
        break

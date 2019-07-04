import glob
import sys
import pyaudio
import wave
import os
import pickle
import numpy as np
import tensorflow as tf
import librosa
from socket import *
from header import *

if len(sys.argv) < 3:
    print("Compile error : python record.py [minutes] [meters]")
    exit(1)

FORMAT = pyaudio.paInt16
NODE = sys.argv[2]
seconds = int(sys.argv[1])

#send node info

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
loop = 0
while loop < seconds:
    try:
        loop = loop+1
        frames = []
        # recording
        for i in range(0, int(RATE/CHUNK*RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        # record/laod wav files
        fileName = file_saver(str(NODE), frames, wave, p)
        # send file & packet
        os.system('scp '+fileName+' gunhoo@192.168.123.6:~/Desktop/Drone-Tracking/server/data/ &')
    # exception handle
    except KeyboardInterrupt:
        print("wait seconds to terminate...")
        stream.stop_stream()
        stream.close()
        p.terminate()
        break


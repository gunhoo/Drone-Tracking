import pyaudio
import librosa
import numpy as np
import tensorflow as tf
from datetime import datetime
########## Variables ##########
RECORD_SECONDS =60 
CHUNK = 8192
CHANNELS = 1
FORMAT = pyaudio.paInt16
RATE = 44100
# Socket Variables
ADDRESS = '192.168.123.6'
PORT = 21536
###############################

########## Functions ##########
def load(frames, sr=RATE):
    [raw, sr] = librosa.load(frames[0], sr=sr)
    for f in frames[1:]:
        [array, sr] = librosa.load(f, sr=sr)
        raw = np.hstack((raw, array))
    return raw
def file_saver(nodeNum, frames, wave, p):
    now = datetime.now()
    time = now.strftime('-%H:%M:%S')
    fileName = './second-'+nodeNum+time+'.wav'
    wf = wave.open(fileName, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return fileName


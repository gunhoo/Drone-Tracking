import glob
import sys
import pyaudio
import wave
import time
import numpy as np
import tensorflow as tf
import librosa
from socket import *

if len(sys.argv) < 2:
    print("Compile error : please input node number ")
    exit(1)

########## Variables ##########
# Recording Variables
RECORD_SECONDS = 1
CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
# Socket Variables
#ADDRESS = '192.168.0.2'
#PORT = 21535
#NODE = int(sys.argv[1])
path = './second.wav'
###############################

########## Functions ##########
def load(frames, sr=RATE):
    [raw, sr] = librosa.load(frames[0], sr=sr)
    for f in frames[1:]:
        [array, sr] = librosa.load(f, sr=sr)
        raw = np.hstack((raw, array))
    return raw
def saver(frames):
    wf = wave.open('second.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
def mfcc4(raw, chunck_size=8192, window_size=4096, sr=RATE, n_mfcc=16, n_frame=16):
    mfcc = np.empty((0, n_mfcc, n_frame))
    for i in range(0, len(raw), chunck_size//2):
        mfcc_slice = librosa.feature.mfcc(raw[i:i+chunck_size], sr=sr, n_mfcc=n_mfcc)
        if mfcc_slice.shape[1] < 17:
            continue
        mfcc_slice = mfcc_slice[:,:-1]
        mfcc_slice = mfcc_slice.reshape((1, mfcc_slice.shape[0], mfcc_slice.shape[1]))
        mfcc = np.vstack((mfcc, mfcc_slice))
    return mfcc
###############################

# connection
#clientSocket = socket(AF_INET, SOCK_DGRAM)
#clientSocket.bind((ADDRESS, PORT))
# send node info
# clinetSocket.send(.encode())

p = pyaudio.PyAudio()

# open pyaudio
stream = p.open(format = FORMAT,
        channels = CHANNELS,
        rate = RATE,
        input = True,
        frames_per_buffer = CHUNK,
        input_device_index = 0,
        output_device_index = 0)

# start loop
print("Start recording...")
sess = tf.Session()
while True:
    try:#
        # initailize values
        
        init = tf.global_variables_initializer()
        sess.run(init)
        frames = []
        # recording
        for i in range(0, int(RATE/CHUNK*RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        # save stream data
        saver(frames)
        # load wav file
        files = glob.glob(path)
        raw_data = load(files)
        # pre-processing
        mfcc_data = mfcc4(raw_data)
        X = np.concatenate(mfcc_data, axis=0)
        ### input data into model
        # model saver
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, './model/model')
        # prediction
        y_pred = sess.run(tf.argmax(logits,1), feed_dict={X:mfcc_data})
        from sklearn.metrics import accuracy_score
        result = (accuracy_score(1, y_pred)*100)%100
        print("result : ", result)
        ### send packet

    # exception handle
    except KeyboardInterrupt:
        print("wait seconds to terminate...")
        break

stream.stop_stream()
stream.close()
p.terminate()



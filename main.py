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
N_MFCC = 16
N_FRAME = 16
N_UNIQ_LABELS = 2

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
def mfcc4(raw, label, chunck_size=8192, window_size=4096, sr=RATE, n_mfcc=16, n_frame=16):
    mfcc = np.empty((0, n_mfcc, n_frame))
    y = []
    for i in range(0, len(raw), chunck_size//2):
        mfcc_slice = librosa.feature.mfcc(raw[i:i+chunck_size], sr=sr, n_mfcc=n_mfcc)
        if mfcc_slice.shape[1] < 17:
            continue
        mfcc_slice = mfcc_slice[:,:-1]
        mfcc_slice = mfcc_slice.reshape((1, mfcc_slice.shape[0], mfcc_slice.shape[1]))
        mfcc = np.vstack((mfcc, mfcc_slice))
        y.append(label)
    y = np.array(y)
    return mfcc
def conv(X, Y):
    # first CNN layer
    conv1 = tf.layers.conv2d(inputs=X,
            filters=1, kernel_size=[3,3],
            padding="SAME", activation=tf.nn.relu)
    # pooling
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
            pool_size=[2, 2], padding="SAME", strides=2)
    # second CNN layer
    conv2 = tf.layers.conv2d(inputs=pool1,
            filters=1, kernel_size=[3, 3],
            padding="SAME", activation=tf.nn.relu)
    # pooling
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
            pool_size=[2, 2], padding="SAME", strides=2)

    flat = tf.reshape(pool2, [-1, 16*16*1])
    dense2 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense2, units=2)
    return logits
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
        mfcc_data, y = mfcc4(raw_data, 1)
        X = np.concatenate(mfcc_data, axis=0)
        y = np.hstack(y)
        n_labels = y.shape[0]
        y_encoded = np.zeros((n_labels, N_UNIQ_LABLES))
        y_encoded[np.arange(n_labels),y] = 1
	# frames 파일 처리 해야하나?
        X = tf.placeholder(tf.float32, shape=[None, N_MFCC*N_FRAME*CHANNELS])
        X = tf.reshape(X, [-1, N_MFCC, N_FRAME, CHANNELS])
        Y = tf.placeholder(tf.float32, shape=[None, N_UNIQ_LABELS])
        # CNN
        logits = conv(X, Y)
        # cost optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
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



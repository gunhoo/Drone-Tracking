import glob
import sys
import pyaudio
import wave
import time
import numpy as np
import tensorflow as tf
import librosa
from socket import *
from header import *

if len(sys.argv) < 2:
    print("Compile error : please input node number ")
    exit(1)

FORMAT = pyaudio.paInt16
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
        #input_device_index = 0,
        #output_device_index = 0)
        )

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
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        # save stream data
        saver(frames, wave, p)
        # load wav file
        files = glob.glob(path)
        raw_data = load(files)
        # pre-processing
        mfcc_data, y = mfcc4(raw_data, 1)
        X = np.concatenate(mfcc_data, axis=0)
        #X_input = X.reshape(X, [-1, N_MFCC, N_FRAME, CHANNELS])
        y = np.hstack(y)
        n_labels = y.shape[0]
        y_encoded = np.zeros((n_labels, N_UNIQ_LABELS))
        y_encoded[np.arange(n_labels),y] = 1
        X = tf.placeholder(tf.float32, shape=[None, N_MFCC*N_FRAME*CHANNELS])
        X = tf.reshape(X, [-1, N_MFCC, N_FRAME, CHANNELS])
        Y = tf.placeholder(tf.float32, shape=[None, N_UNIQ_LABELS])
        # CNN
        logits = conv(X, Y)
        # cost optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
        # model saver
        sess = tf.Session()
        saver = tf.train.import_meta_graph('./model/model.meta')
        saver.restore(sess, './model/model')
        # prediction
        y_pred = sess.run(tf.argmax(logits,1), feed_dict={X:X})
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



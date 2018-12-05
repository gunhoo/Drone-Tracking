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
    clientSocket.connect((ADDRESS,PORT))
except Exception as e:
    print('cannot connect to the server;', e)
    exit()
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

        # pre-processing
        mfcc_data, y = mfcc4(raw_data, 1)
        printer("MFCC")
        X = np.concatenate((mfcc_data), axis=0)
        X_input = X.reshape(-1,N_MFCC,N_FRAME,CHANNELS)
        y = np.hstack(y)
        n_labels = y.shape[0]
        y_encoded = np.zeros((n_labels, N_UNIQ_LABELS))
        y_encoded[np.arange(n_labels),y] = 1
        X = tf.placeholder(tf.float32, shape=[None, N_MFCC*N_FRAME*CHANNELS])
        X = tf.reshape(X, [-1, N_MFCC, N_FRAME, CHANNELS])
        Y = tf.placeholder(tf.float32, shape=[None, N_UNIQ_LABELS])

        # CNN layer
        logits = conv(X)
        # cost optimizer needed??? -> time consuming
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
        printer("layer")

        # model saver
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, './model/CNN/cnn_model')
        printer("Model saver")

        # prediction
        y_pred = sess.run(tf.argmax(logits,1), feed_dict={X:X_input})
        #y_true = sess.run(tf.argmax(y_encoded,1))
        from sklearn.metrics import accuracy_score
        result = "%2.2f" %((accuracy_score(y, y_pred)*100)%100)
        printer(result)

        ### send packet
        message = NODE + ":" + str(result) + ":" + posX + ":" + posY
        clientSocket.send(message.encode())
        printer("TCP")
    # exception handle
    except KeyboardInterrupt:
        print("wait seconds to terminate...")
        stream.stop_stream()
        stream.close()
        p.terminate()
        clientSocket.close()
        break

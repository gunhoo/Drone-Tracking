import pyaudio
import librosa
import numpy as np
import tensorflow as tf
########## Variables ##########
# Recording Variables
RECORD_SECONDS = 2
CHUNK = 8192
CHANNELS = 1
FORMAT = pyaudio.paInt16
RATE = 44100
N_MFCC = 16
N_FRAME = 16
N_UNIQ_LABELS = 2
LEARNING_RATE = 0.0002
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
def file_saver(frames, wave, p):
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
    return mfcc, y
def conv(X, Y):
    # first CNN layer
    conv1 = tf.layers.conv2d(inputs=X,
            filters=1, kernel_size=[3,3],
            padding="SAME", activation=tf.nn.relu)
    # pooling
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
            pool_size=[2, 2], padding="SAME", strides=1)
    # second CNN layer
    conv2 = tf.layers.conv2d(inputs=pool1,
            filters=1, kernel_size=[3, 3],
            padding="SAME", activation=tf.nn.relu)
    # pooling
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
            pool_size=[2, 2], padding="SAME", strides=1)

    flat = tf.reshape(pool2, [-1, 16*16*1])
    dense2 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense2, units=2)
    return logits
###############################

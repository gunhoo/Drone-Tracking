import threading
from socket import *
import sys
import os
import pyaudio
import wave
import glob
import tensorflow as tf
import numpy as np
from datetime import datetime
from header import *

if len(sys.argv) < 2:
    print("compile error: please input total node number")
    exit(1)
totalNodeNum = int(sys.argv[1])
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'
info = []
posX = []
posY = []
for i in range(0,totalNodeNum):
    info.append(0)
    posX.append(-1)
    posY.append(-1)

class ClientThread(threading.Thread):
    def __init__(self, clientAddress, connectionSocket):
        threading.Thread.__init__(self)
        self.csocket = connectionSocket
        self.caddr = clientAddress
    def cal(self, nodeNum):
        global info, totalNodeNum
        predX = 0
        predY = 0
        tmp = 0
        for i in range(0, totalNodeNum):
            predX += int(info[i])*int(posX[i])
            predY += int(info[i])*int(posY[i])
            tmp += int(info[i])
        if tmp != 0:
            predX = predX / tmp
            predY = predY / tmp
        else:
            predX = 'not found'
            predY = 'not found'
        now = datetime.now()
        time = now.strftime('%H:%M:%S:%f')
        print("-----",nodeNum,">",time,": Drone's location: (", predX, ",", predY, ")-----")
    def run(self):
        global info, posX, posY
        print("*****Client Address", self.caddr[0], "connected.*****")
        message = self.csocket.recv(1024).decode()
        modifiedMessage = message.split(':')
        nodeNum = int(modifiedMessage[0])
        posX[nodeNum] = int(modifiedMessage[1])
        posY[nodeNum] = int(modifiedMessage[2])
        while True:
            try:
                with tf.device("/gpu:0"):
                    start_t = datetime.now()
                    sess = tf.Session()
                    init = tf.global_variables_initializer()
                    tf.reset_default_graph()
                    sess.run(init)
                    printer(str(nodeNum)+">Start")
                    fileName = self.csocket.recv(35).decode()
                    printer(str(nodeNum)+">socket receive")
                    while not os.path.exists(fileName):
                        continue
                    while os.path.getsize(fileName)/1024 < 30:
                        continue
                    printer(str(nodeNum)+">file receive")
                    files = glob.glob(fileName)
                    raw_data = load(files)
                    printer(str(nodeNum)+">file load")
                    
                    # pre-processing
                    mfcc_data, y_data = mfcc4(raw_data,4)
                    print("mfcc",mfcc_data.shape, y_data.shape)
                    printer(str(nodeNum)+">MFCC")
                    X = np.concatenate((mfcc_data), axis=0)
                    X = X.reshape(-1, N_MFCC, N_FRAME, CHANNELS)
                    X_input = np.reshape(X,(X.shape[0],-1))
                    y = np.hstack((y_data))
                    n_labels = y.shape[0]
                    y_encoded = np.zeros((n_labels, N_UNIQ_LABELS))
                    y_encoded[np.arange(n_labels),y] = 1

                    tf.reset_default_graph()

                    X = tf.placeholder(tf.float32, shape=[None,N_MFCC*N_FRAME*CHANNELS])
                    Y = tf.placeholder(tf.float32, shape=[None,N_UNIQ_LABELS])
                    keep_prob = tf.placeholder(tf.float32)
                    # Dense layer
                    logits = dens(X, keep_prob)
                    printer(str(nodeNum)+">Layer")
                    
                    # cost optimizer needed??? -> time consuming
                    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
                    #optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
                    #printer(str(nodeNum)+">cost-optimizer")
                    
                    # model saver
                    sess = tf.Session()
                    saver = tf.train.Saver()
                    saver.restore(sess, '../model/Dense/4_dense_model')
                    printer(str(nodeNum)+">Model saver")
                    
                    # prediction
                    y_pred = sess.run(tf.argmax(logits,1),feed_dict={X:X_input,keep_prob:1})
                    y_true = sess.run(tf.argmax(y_encoded,1))
                    from sklearn.metrics import confusion_matrix
                    mat = confusion_matrix(y_true, y_pred)
                    print(mat)
                    print(len(mat))
                    distance = 0
                    for i in range(1, len(mat)):
                        distance = distance + mat[0][i]*50/i
                    #distance = (mat[0][1]*50+mat[0][2]*20+mat[0][3]*10+mat[0][4]) / (mat[0][0]+mat[0][1]+mat[0][2]+mat[0][3]+mat[0][4])
                    printer(distance)
                    info[nodeNum] = distance
                    self.cal(nodeNum)
                    end_t = datetime.now()
                    print('one cycle : ', end_t - start_t)
            except KeyboardInterrupt:
                self.csocket.close()
                exit()

PORT = 21536
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind((ADDRESS, PORT))

print("The server is ready to receive on port", PORT)

while True:
    try:
        serverSocket.listen(1)
        (connectionSocket, clientAddress) = serverSocket.accept()
        newThread = ClientThread(clientAddress, connectionSocket)
        newThread.start()
    except KeyboardInterrupt:
        connectionSocket.close()
        print('Bye bye~')
        exit()
exit()

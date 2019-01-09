import threading
from socket import *
import sys
from datetime import datetime
from header import *

if len(sys.argv) < 2:
    print("compile error: please input total node number")
    exit(1)
totalNodeNum = int(sys.argv[1])
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
            predX += (info[i]*posX[i])
            predY += (info[i]*posY[i])
            tmp += info[i]
        if tmp != 0:
            predX = predX / tmp
            predY = predY / tmp
        else:
            predX = 'not found'
            predY = 'not found'
        now = datetime.now()
        time = now.strftime('%H:%M:%S')
        print(nodeNum,">", time,": Drone's location: (", predX, ",", predY, ")")
    def run(self):
        global info, posX, posY
        print("Client Address", self.caddr[0], "connected.")
        while True:
            try:
                message = self.csocket.recv(2048).decode()
                modifiedMessage = message.split(':')
                nodeNum = int(modifiedMessage[0])
                percentage = int(modifiedMessage[1])
                posX[nodeNum] = int(modifiedMessage[2])
                posY[nodeNum] = int(modifiedMessage[3])
                info[nodeNum] = percentage
                self.cal(nodeNum)
            except KeyboardInterrupt:
                self.csocket.close()

PORT = 21535
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

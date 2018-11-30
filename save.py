import pyaudio
import time
import datetime
import wave
import numpy as np 
import sys

def getStream(sample_rate = 44100, chunk_size = 8192,chunk_num = 10, isWrite=False):  
   AUDIO_FORMAT = pyaudio.paInt16
   SAMPLE_RATE = sample_rate
   CHUNK_SIZE = chunk_size
   CHUNK_NUM = chunk_num
   p = pyaudio.PyAudio()
       
   
 
   while(True):
       WAVE_FILENAME = './'+datetime.datetime.now().strftime('%m-%d %H_%M_%S')+'.wav'
       stream = p.open(format=AUDIO_FORMAT, channels=1, rate=SAMPLE_RATE,
        input=True, frames_per_buffer=CHUNK_SIZE)#, input_device_index=0, output_device_index =0 )
        
       frame = []  
       t1 = time.time()
       cn = 0
       for i in range(CHUNK_NUM *20):
           frame.append(stream.read(CHUNK_SIZE,exception_on_overflow = False))
           cn+=1
           
       frame = b''.join(frame)
       audio = np.fromstring(frame, np.int16)
       
       t2 = time.time()
       
       # write to the audio file
       wf = wave.open(WAVE_FILENAME, 'wb')
       wf.setnchannels(1)
       wf.setsampwidth(p.get_sample_size(AUDIO_FORMAT))
       wf.setframerate(SAMPLE_RATE)
       wf.writeframes(b''.join(audio))
       print("time: %.4f \t"%(t2-t1),end='')
       stream.stop_stream()
       stream.close()

getStream()

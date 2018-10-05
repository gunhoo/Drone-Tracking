import pyaudio
import wave
import sys

if (len(sys.argv) < 3):
    print('compile example : python3 record.py [record time(s)] [output file name.wav]')
    sys.exit(0)

WAVE_OUTPUT_FILENAME = sys.argv[2]
RECORD_SECONDS = int(sys.argv[1])
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()

stream = p.open(format = FORMAT,
                channels = CHANNELS,
                rate = RATE,
                input = True,
                frames_per_buffer = CHUNK,input_device_index=0, output_device_index =0)

print("start to record the audio.")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
    
print("Recording finished.")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

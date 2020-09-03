import pyaudio
import numpy as np

class Speaker:
    def __init__(self, sampling_rate=48000):
        self.SAMPLING_RATE = sampling_rate
        
        self.speaker = pyaudio.PyAudio()
        
    def say(self, data):
        self.stream = self.speaker.open(format=pyaudio.paInt16, channels=1, rate=self.SAMPLING_RATE, output=True)

        self.stream.write(data.tobytes())

        self.stream.stop_stream()
        self.stream.close()

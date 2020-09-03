import pyaudio
import numpy as np

class Listener:
    def __init__(self, feature_length=0.16, chunks_per_feature=4, sampling_rate=16000):
        self.FEATURE_LENGTH = feature_length
        self.CHUNKS_PER_FEATURE = chunks_per_feature
        self.SAMPLING_RATE = sampling_rate
        self.CHUNK_LENGTH = self.FEATURE_LENGTH / self.CHUNKS_PER_FEATURE
        self.SAMPLES_PER_CHUNK = int(self.CHUNK_LENGTH * self.SAMPLING_RATE)
        
        self.microphone = pyaudio.PyAudio()

    def listen(self):
        self.stream = self.microphone.open(format=pyaudio.paInt16, channels=1, rate=self.SAMPLING_RATE, input=True, frames_per_buffer=self.SAMPLES_PER_CHUNK)
        
        STARTBUFFER, STOPBUFFER = 2 * self.CHUNKS_PER_FEATURE, 2 * self.CHUNKS_PER_FEATURE
        start, stop, recorded = 0, 0, [[0 for i in range(self.SAMPLES_PER_CHUNK * 2)] for j in range(STARTBUFFER)]

        while True:
            data = self.stream.read(self.SAMPLES_PER_CHUNK)
            data = np.frombuffer(data, dtype='int16')
            recorded.append(list(data))

            try:
                if max(recorded[-1]) >= 120 and all([max(recorded[-i]) <= 120 for i in range(2, STARTBUFFER)]) and not(start):
                    start = len(recorded) - STARTBUFFER
            
                elif all([max(recorded[-i]) <= 120 for i in range(2, STOPBUFFER)]) and max(recorded[-STOPBUFFER]) >= 120 and start:
                    self.stream.stop_stream()
                    self.stream.close()
                    
                    stop = len(recorded)

                    active = []
                    for chunk in recorded[start:stop]:
                        active += chunk
                    active = np.array(active).astype('int16')

                    return active
            
            except:
                pass

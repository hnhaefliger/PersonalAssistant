from Audio import Output
from TTS import Synthesis
import re

if __name__ == '__main__':
    speaker = Output.Speaker()
    
    regex = re.compile('[^a-zA-Z ]')
    
    with open('Dataset/TTS/dictionary.txt', 'r') as f:
        data = f.read().split('\n')[:-1]
        data = [line.split('  ') for line in data]
        data = [[line[0], regex.sub('', line[1]).split(' ')] for line in data if len(line) == 2]
        data = {line[0]: line[1] for line in data}

    phonemes = []

    for word in 'this is a demo'.upper().split(' '):
        phonemes += data[word] + [' ']

    Synthesis.synthesize(phonemes, 'Dataset/TTS/male', 'Dataset/TTS')
    
    audio = Synthesis.getAudio('Dataset/TTS/output.wav')

    speaker.say(audio)

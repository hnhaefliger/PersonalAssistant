import STT.Model, TTS.Model, TTS.Synthesis
from Audio import Input, Output
import Data.STT.Utils, Data.TTS.Utils
from Command import Commands
from Command import Functions

'''
import re

regex = re.compile('[^a-zA-Z ]')
    
with open('Dataset/TTS/dictionary.txt', 'r') as f:
    matches = f.read().split('\n')[:-1]
    matches = [line.split('  ') for line in matches]
    matches = [[line[0], regex.sub('', line[1]).split(' ')] for line in matches if len(line) == 2]
    matches = {line[0]: line[1] for line in matches}
'''

if __name__ == '__main__':
    listenerModel = STT.Model.Model(loadfrom='STT/Models/model1.h5')
    listenerModel.segmentModel()

    listener = Input.Listener()

    handler = Commands.Handler('Command/Commands.txt')

    synthesisModel = TTS.Model.Model(loadfrom='TTS/Models/model1.h5')
    synthesisModel.segmentModel()

    speaker = Output.Speaker()

    while True:
        print('recording')
        
        data = listener.listen()

        print('done')
        
        data = Data.STT.Utils.createSpectrogram(data)

        data = Data.STT.Utils.padSpectrogram(data, len(data) + 8 - len(data) % 8)

        data = listenerModel.predict(data)

        data = Data.STT.Utils.arrayToString(data)

        print(data)

        data = handler.match(data)

        if data:
            data = eval('Functions.' + data)

            print(data)

            phones = []

            for word in data.split(' '):
                data = Data.TTS.Utils.stringToArray(word)

                phones += Data.TTS.Utils.arrayToPhones(synthesisModel.predict(data)).split(' ') + [' ']
            
            '''
            for word in data.upper().split(' '):
                phones += matches[word] + [' ']
            '''

            TTS.Synthesis.synthesize(phones, 'Dataset/TTS/male', 'Data/TTS')
            
            speaker.say(TTS.Synthesis.getAudio('Data/TTS/output.wav'))

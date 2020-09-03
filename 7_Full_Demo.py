import STT.Model, TTS.Model, TTS.Synthesis
from Audio import Input, Output
from Data import STT.Utils, TTS.Utils
from Command import Commands
from Command import Functions

if __name__ = '__main__':
    listenerModel = STT.Model.Model(loadfrom='STT/Models/model1.h5')
    listenerModel.segmentModel()

    listener = Input.Listener()

    handler = Commands.Handler('Command/Commands.txt')

    synthesisModel = TTS.Model.Model(loadfrom='TTS/Model/model1.h5')
    synthesisModel.segmentModel()

    speaker = Output.Speaker()

    while True:
        data = listener.listen()
        
        data = STT.Utils.createSpectrogram(data)

        data = STT.Utils.padSpectrogram(data, len(data) + 8 - len(data) % 8)

        data = listenerModel.predict(data)

        data = STT.Utils.arrayToString(data)

        data = handler.match(data)

        data = eval('Function.' + data)

        phones = []

        for word in data.split(' '):
            data = TTS.Utils.stringToArray(word)

            phones += TTS.Utils.arrayToPhones(synthesisModel.predict(data)).split(' ') + [' ']

        TTS.Synthesis.synthesize(phones, 'Dataset/TTS/male', 'Data/TTS')
        
        speaker.say(TTS.Synthesis.getAudio('Data/TTS/output.wav'))

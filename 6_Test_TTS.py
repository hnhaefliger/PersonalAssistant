from TTS.Model import Model
from Data.TTS import Utils
from TTS import Synthesis
from Audio import Output

if __name__ == '__main__':
    model = Model(loadfrom='TTS/Models/model1.h5')
    model.segmentModel()

    speaker = Output.speaker()

    phones = []

    for word in 'this is a demo'.split(' '):
        data = Utils.stringToArray(word)

        phones += Utils.arrayToPhones(model.predict(data)).split(' ') + [' ']

    Synthesis.synthesize(phones, 'Dataset/TTS/male', 'Data/TTS')
    
    speaker.say(Synthesis.getAudio('Data/TTS/output.wav'))

    

from STT import Model
from Audio import Input
from Data.STT import Utils

if __name__ == '__main__':
    model = Model.Model(loadfrom='STT/Models/model1.h5')
    model.segmentModel()

    listener = Input.Listener()

    while True:
        print('recording')
        
        data = listener.listen()

        print('done')

        data = Utils.createSpectrogram(data)

        data = Utils.padSpectrogram(data, len(data) + 8 - len(data) % 8)

        data = model.predict(data)

        data = Utils.arrayToString(data)

        print(data)


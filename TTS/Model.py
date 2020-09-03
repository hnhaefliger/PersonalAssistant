from tensorflow.keras.models import Model as model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Bidirectional, TimeDistributed, Attention, Concatenate
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
from Data.TTS import Utils

class Model:
    '''
    Class to define and train the speech recognition model
    '''
    
    def __init__(self, summary=False, loadfrom=False):
        '''
        Define an encoder-decoder model
        '''

        if loadfrom:
            self.model = load_model(loadfrom, custom_objects={'Attention': Attention})

        else:
            encoder_input = Input(shape=(None, 29))
            decoder_input = Input(shape=(None, 43))

            # bLSTM
            
            encoder = Bidirectional(LSTM(256, return_sequences=True, return_state=False))(encoder_input)

            # define the encoder

            encoder, state_fh, state_fc, state_bh, state_bc = Bidirectional(LSTM(256, return_sequences=True, return_state=True), merge_mode='sum')(encoder)
            states = [state_bh, state_bc]

            # define the decoder

            decoder, state_h, state_c = LSTM(256, return_sequences=True, return_state=True)(decoder_input, initial_state=states)

            # calculate Luong attention based on the encoder and decoder outputs

            attention = Attention()([decoder, encoder])

            inner = Concatenate()([attention, decoder])

            # output layer is a one-hot array with softmax activation

            output = TimeDistributed(Dense(43, activation='softmax'))(inner)

            # define the model
                
            self.model = model(inputs=[encoder_input, decoder_input], outputs=output)

        if summary:
            self.model.summary()

    def fit(self, dataset, epochs=1, optimizer='adam', loss='categorical_crossentropy'):
        '''
        Fit the model to a dataset
        '''
        
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return self.model.fit(dataset, epochs=epochs, steps_per_epoch=len(dataset), verbose=1, workers=1, max_queue_size=10, use_multiprocessing=False)

    def save(self, location):
        '''
        Save the model to a given location
        '''
        
        self.model.save(location)

    def segmentModel(self):
        '''
        Split the model into encoder/decoder/output sections
        '''

        # generate encoder module
        encoder_input = Input(shape=(None, 29))
        encoder = self.model.layers[1](encoder_input)
        encoder, state_fh, state_fc, state_bh, state_bc = self.model.layers[3](encoder)
        
        self.encoder = model(inputs=encoder_input, outputs=[encoder, state_bh, state_bc])
        self.encoder.compile(optimizer='adam', loss='categorical_crossentropy')

        # generate decoder module
        decoder_input = Input(shape=(None, 43))
        decoder_state_h = Input(shape=(256,))
        decoder_state_c = Input(shape=(256,))
        decoder, state_h, state_c = self.model.layers[4](decoder_input, initial_state=[decoder_state_h, decoder_state_c])
        self.decoder = model(inputs=[decoder_input, decoder_state_h, decoder_state_c], outputs=[decoder, state_h, state_c])
        self.decoder.compile(optimizer='adam', loss='categorical_crossentropy')

        # generate attention + output layers
        output_decoder = Input(shape=(None, 256))
        output_encoder = Input(shape=(None, 256))

        attention = self.model.layers[5]([output_decoder, output_encoder])
        output = self.model.layers[6]([attention, output_decoder])
        output = self.model.layers[7](output)
        self.output = model(inputs=[output_decoder, output_encoder], outputs=output)
        self.output.compile(optimizer='adam', loss='categorical_crossentropy')

    def predict(self, x):
        x = np.array([x])
        
        encoding, state_h, state_c = self.encoder.predict(x)

        y = np.array([Utils.phonesToArray('\t')])

        while not(Utils.checkPhonemesEnd(y[0][-1])):
            decoding, state_h, state_c = self.decoder.predict([np.array([y[:, -1]]), state_h, state_c])

            pred = self.output.predict([decoding, encoding])

            y = np.concatenate((y, pred), axis=1)

        return y[0][1:-1]

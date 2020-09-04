from tensorflow.keras.utils import Sequence
import numpy as np
from Data.TTS import Utils
import re

class Dataset(Sequence):
    def __init__(self, location, model, batch_size=8, interval=[0,-1], teacher_forcing=lambda x: not(x % 5)):
        '''
        Initialize dataset by fetching sentence/recording pairs
        '''

        self.model = model
        self.teacher_forcing = teacher_forcing
        
        self.location = location
        self.batch_size = batch_size

        regex = re.compile('[^a-zA-Z ]')

        with open('Dataset/TTS/dictionary.txt', 'r') as f:
            data = f.read().split('\n')[:-1]
            data = [line.split('  ') for line in data]
            data = [[line[0], regex.sub('', line[1]).split(' ')] for line in data if len(line) == 2 and Utils.allAlpha(line[0])]
            
        self.x1 = [line[0].lower() for line in data]
        self.x2 = [['\t'] + line[1] for line in data]
        self.y = [line[1] + ['\n'] for line in data]

    def __len__(self):
        '''
        Get length of dataset to calculate number of batches per epoch
        '''

        return int(len(self.x1) / self.batch_size)

    def __getitem__(self, idx):
        '''
        Get one batch of data
        - Convert recordings to spectrograms
        - Convert sentences to one-hot arrays
        '''
        
        x1 = self.x1[self.batch_size*idx:self.batch_size*(idx+1)]
        x2 = self.x2[self.batch_size*idx:self.batch_size*(idx+1)]
        y = self.y[self.batch_size*idx:self.batch_size*(idx+1)]

        x1_length = [len(word) for word in x1]
        x1_padding = max(x1_length)

        x1 = np.array([Utils.stringToArray(word, padding=x1_padding) for word in x1])

        x2_length = [len(phones) for phones in x2]
        x2_padding = max(x2_length)

        x2 = np.array([Utils.phonesToArray(phones, padding=x2_padding) for phones in x2])

        if self.teacher_forcing(idx):
            x2 = np.array(self.model.model.predict((x1, x2)))
            x2 = np.concatenate((np.array([Utils.phonesToArray('\t') for i in range(self.batch_size)]), x2), axis=1)
        
        y = np.array([Utils.phonesToArray(phones, padding=x2_padding) for phones in y])

        return (x1, x2), y        

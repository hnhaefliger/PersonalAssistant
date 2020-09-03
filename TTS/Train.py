from TTS import Model
from Data.TTS import Dataset

class Trainer:
    def __init__(self, dataset, batch_size=8, interval=[0,-1], loadfrom=False):
        '''
        Prepare model for training
        '''

        self.model = Model.Model(loadfrom=loadfrom)

        self.dataset = Dataset.Dataset(dataset, batch_size=batch_size, interval=interval)

    def reloadDataset(self, dataset, batch_size=8, interval=[0,-1]):
        '''
        Change dataset (for training with segments of the dataset
        '''
        
        self.dataset = Dataset.Dataset(dataset, batch_size=batch_size, interval=interval)

    def train(self, epochs):
        '''
        Train the model
        '''

        return self.model.fit(self.dataset, epochs=epochs)

    def saveModel(self, location):
        '''
        Save the model
        '''

        self.model.save(location)

        

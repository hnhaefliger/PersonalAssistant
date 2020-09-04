from STT import Model
from Data.STT import Dataset

class Trainer:
    def __init__(self, dataset, batch_size=8, interval=[0,-1], loadfrom=False, teacher_forcing=lambda x: not(x % 5)):
        '''
        Prepare model for training
        '''

        self.model = Model.Model(loadfrom=loadfrom)
        self.teacher_forcing = teacher_forcing

        self.dataset = Dataset.Dataset(dataset, self.model, batch_size=batch_size, interval=interval, teacher_forcing=self.teacher_forcing)

    def reloadDataset(self, dataset, batch_size=8, interval=[0,-1]):
        '''
        Change dataset (for training with segments of the dataset
        '''
        
        self.dataset = Dataset.Dataset(dataset, self.model, batch_size=batch_size, interval=interval, teacher_forcing=self.teacher_forcing)

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

        

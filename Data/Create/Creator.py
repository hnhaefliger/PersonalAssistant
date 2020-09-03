from Audio import Input, Output
import os
import wave

def initDataset(location):
    '''
    Set up location for dataset to be stored
    - Create a file to store sentence/recording pairs
    - Create a sub-directory to store recordings
    '''
    
    file = open(location + '/pairs.csv', 'w+')
    file.close()

    try:
        os.mkdir(location + '/recordings')

    except FileExistsError:
        pass

class Creator:
    '''
    Class to accelerate the creation of a custom dataset by recording a user speaking
    '''
    
    def __init__(self, sentences, dataset):
        '''
        Initialize a recording class and a playback class for interaction with the user
        Locate dataset to resume from previous recording
        Gather sentences to record
        '''
        
        self.listener = Input.Listener()
        self.speaker = Output.Speaker(sampling_rate=16000)

        try:
            with open(dataset + '/pairs.csv', 'r') as f:
                self.current_sentence = int(f.read().split('\n')[-2].split(',')[0].replace('.wav', ''))

        except:
            initDataset(dataset)
            self.current_sentence = 0

        with open(sentences, 'r') as f:
            self.sentences = f.read().split('\n')[self.current_sentence:-1]

        self.dataset = dataset

    def nextSentence(self):
        '''
        Record the user speaking and store the sentence/recording pair in the dataset directory
        '''
        
        print(self.sentences[0])
        self.current_sentence += 1
        
        if input('press [enter] to begin recording or [s] to skip this sentence ') != 's':
            print('started recording')
            
            data = self.listener.listen()

            print('done')

            while True:
                cont = input('press [enter] to save and contine, [r] to listen to the recording or [n] to record again ')

                if cont == 'n':
                    print('started recording')
                    
                    data = self.listener.listen()

                    print('done')

                elif cont == 'r':
                    self.speaker.say(data)

                else:
                    file = wave.open(self.dataset + '/recordings/' + str(self.current_sentence) + '.wav', 'wb')
                    
                    file.setparams((1, 2, 16000, 0, 'NONE', ''))

                    file.writeframes(data)

                    file.close()

                    with open(self.dataset + '/pairs.csv', 'a+') as f:
                        f.write(str(self.current_sentence) + '.wav,"' + self.sentences[0] + '"\n')
                    
                    break
                
        print('')
        del self.sentences[0]

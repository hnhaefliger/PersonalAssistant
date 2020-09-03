import wave
from scipy.signal import spectrogram
import math
import numpy as np

chars = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split() + [' ', '\t', '\n']

char_map = {chars[i]: i for i in range(len(chars))}
index_map = {i: chars[i] for i in range(len(chars))}

def createSpectrogram(audio, samplerate=16000, winlen=0.01, nperseg=160):
    '''
    Generate and array of spectrograms with specified window sizes
    '''
    
    winsamples = int(samplerate * winlen)
    padding = np.zeros(winsamples - (len(audio) % winsamples))
    audio = np.concatenate((audio, padding))
    
    features = np.array([audio[i * winsamples:(i+1) * winsamples] for i in range(int(len(audio) / winsamples))])
    features = np.array([spectrogram(f, samplerate, nperseg=nperseg)[2] for f in features])
    features = features.reshape((features.shape[:2]))
    
    return features

def padSpectrogram(features, padding):
    '''
    Add padding to a sequence of spectrograms
    '''
    
    if len(features) > padding:
        raise ValueError("The length of the audio cannot exceed the padded length")
        
    pad = np.array([np.zeros(features.shape[1:])])
        
    for i in range(padding - len(features)):
        features = np.concatenate((features, pad))

    return features

def fileToSpectrogram(file, samplerate=16000, padding=False, winlen=0.01, nperseg=160):
    '''
    Create spectrograms from file with padding for training
    '''
    
    audio = wave.open(file, 'rb')
    audio = audio.readframes(audio.getnframes())
    audio = np.frombuffer(audio, dtype=np.int16)
    
    features = createSpectrogram(audio, samplerate=samplerate, winlen=winlen, nperseg=nperseg)

    if padding:
        features = padSpectrogram(features, padding)
        
    return features

def maxSpectrogramLength(file, samplerate=16000, winlen=0.01):
    '''
    Calculate number of frames for a given file for padding of other files in batch
    '''
    
    audio = wave.open(file, 'rb')
    audio = audio.readframes(audio.getnframes())
    audio = np.frombuffer(audio, dtype=np.int16)
    
    winsamples = int(samplerate * winlen)
    
    length = math.ceil(len(audio) / winsamples) + 8
    length -= length % 8
    
    return length
    
def stringToArray(string, padding=False):
    '''
    Convert a given string to a one-hot array encoding
    '''
    
    indexes = [char_map[letter] for letter in string]
    array = []
    
    for letter in indexes:
        zeros = np.zeros(len(chars))
        zeros[letter] = 1
        array.append(zeros)

    if padding:
        if len(array) > padding:
            raise ValueError("The length of a string cannot exceed the padded length")
        
        pad = np.zeros(len(chars))
        pad[char_map['\n']] = 1
        
        for i in range(padding - len(array)):
            array.append(pad)

    return np.array(array)
        
def arrayToString(array):
    '''
    Convert a sequence of one-hot arrays into a string
    '''
    
    indexes = [np.argmax(letter) for letter in array]

    string = [index_map[letter] for letter in indexes]
    string = ''.join(string)

    return string

def checkEnd(array):
    '''
    Check if a one-hot array is the <EOS> character
    '''

    if index_map[np.argmax(array)] == '\n':
        return True

    return False
    

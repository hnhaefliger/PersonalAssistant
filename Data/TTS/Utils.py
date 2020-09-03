import numpy as np

chars = 'a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z, ,\t,\n'.split(',')

char_map = {chars[i]: i for i in range(len(chars))}
cindex_map = {i: chars[i] for i in range(len(chars))}

phones = 'B,K,D,F,G,HH,JH,L,M,N,P,R,S,T,V,W,Y,Z,NG,ZH,CH,SH,TH,DH,AE,P,EH,IH,AA,AH,UH,EY,IY,AY,OW,UW,AW,OY,AO,ER, ,\t,\n'.split(',')

phone_map = {phones[i]: i for i in range(len(phones))}
pindex_map = {i: phones[i] for i in range(len(phones))}

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

    string = [cindex_map[letter] for letter in indexes]
    string = ''.join(string)

    return string

def checkStringEnd(array):
    '''
    Check if a one-hot array is the <EOS> character
    '''

    if cindex_map[np.argmax(array)] == '\n':
        return True

    return False

def phonesToArray(string, padding=False):
    '''
    Convert sequence of phonemes to one hot arrays
    '''

    if type(string) == 'string':
        indexes = [phone_map[phone] for phone in string.split(' ')]
        
    else:
        indexes = [phone_map[phone] for phone in string]
        
    array = []
    
    for phone in indexes:
        zeros = np.zeros(len(phones))
        zeros[phone] = 1
        array.append(zeros)

    if padding:
        if len(array) > padding:
            raise ValueError("The length of phones cannot exceed the padded length")
        
        pad = np.zeros(len(phones))
        pad[phone_map['\n']] = 1
        
        for i in range(padding - len(array)):
            array.append(pad)

    return np.array(array)

def arrayToPhones(array):
    '''
    Convert a sequence of one-hot arrays into phonemes
    '''
    
    indexes = [np.argmax(phone) for phone in array]

    string = [pindex_map[phone] for phone in indexes]
    string = ' '.join(string)

    return string

def checkPhonesEnd(array):
    '''
    Check if a one-hot array is the <EOS> character
    '''
    
    if pindex_map[np.argmax(array)] == '\n':
        return True

    return False

def allAlpha(string):
    '''
    check if string is a word
    '''
    
    return all([l.isalpha() for l in string])

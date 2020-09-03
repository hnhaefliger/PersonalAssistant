import wave
import numpy as np

phones_to_files = '''B	b.wav
K	k.wav
D	d.wav
F	f.wav
G	g.wav
HH	h.wav
JH	j.wav
L	l.wav
M	m.wav
N	n.wav
P	p.wav
R	r.wav
S	s.wav
T	t.wav
V	v.wav
W	w.wav
Y	y.wav
Z	z.wav
NG	ng.wav
ZH	zh.wav
CH	ch.wav
SH	sh.wav
TH	a_th.wav
DH	b_th.wav
AE	a.wav
P	p.wav
EH	e.wav
IH	i.wav
AA	o.wav
AH	u.wav
UH	oo.wav
EY	ai.wav
IY	ee.wav
AY	ie.wav
OW	oa.wav
UW	ew.wav
AW	ou.wav
OY	oy.wav
AO	au.wav
ER	er.wav
 	space.wav'''.split('\n')

phones_to_files = [line.split('\t') for line in phones_to_files]
phones_to_files = {line[0]: line[1] for line in phones_to_files}

def synthesize(phones, phones_location, save_location):
    '''
    Combine the files corresponding to a sequence of phonemes into a single file
    '''
    
    files = [wave.open(phones_location + '/' + phones_to_files[phone], 'rb') for phone in phones]
    audio = [[file.getparams(), file.readframes(file.getnframes())] for file in files]

    for file in files:
        file.close()

    output = wave.open(save_location + '/output.wav', 'wb')
    output.setparams(audio[0][0])

    for value in audio:
        output.writeframes(value[1])

    output.close()

def getAudio(location):
    '''
    Return the audio from the synthesized file for playback
    '''
    
    file = wave.open(location, 'rb')
    
    audio = file.readframes(file.getnframes())
    audio = np.frombuffer(audio, dtype=np.int16)

    file.close()

    return audio
        

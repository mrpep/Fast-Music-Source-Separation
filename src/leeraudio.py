import pydub
import os
import scipy.io.wavfile as wavfile
from shutil import copyfile
import numpy as np

def ReadAudio(filename):
    extension = filename.split('.')
    extension = extension[-1]
    if extension == 'mp3' or extension == 'Mp3' or extension == 'MP3':
    
    #Abrir mp3:
        mp3 = pydub.AudioSegment.from_mp3(filename)
        mp3.export("tempaudio.wav",format="wav")
    
    elif extension == 'ogg':
        mp3 = pydub.AudioSegment.from_ogg(filename)
        mp3.export("tempaudio.wav",format="wav")
   
    elif extension == 'wav':
        copyfile(filename,"tempaudio.wav")

    [fs,audiosignal] = wavfile.read("tempaudio.wav")
    os.remove("tempaudio.wav")

    return fs, audiosignal
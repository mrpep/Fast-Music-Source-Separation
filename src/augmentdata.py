# =============================================================================
# augmentdata.py - Leonardo Pepino (Universidad Nacional de Tres de Febrero)
#
# This script defines data augmentation routines to apply during the training
# phase.
# =============================================================================

from pysndfx import AudioEffectsChain
import scipy.io.wavfile as wavfile
import numpy as np

"""Tecnicas de Data Augmentation propuestas en: Improving music source separation
 based on deep neural networks through data augmentation and network blending - 
 Stefan Uhlich, Marcello Porcu, Franck Giron, Michael Enenkl, Thomas Kemp, 
 Naoya Takahashi y Yuki Mitsufuji.
"""

def changeamplitudes(instruments):

    """ Devuelve las pistas remezcladas con nuevas ganancias y la mezcla 
    resultante."""
    
    minimo = 0.25
    maximo = 1.25
    newsources = []
    mix = 0
    for instrument in instruments:
        amplitude = np.random.uniform(minimo,maximo)
        instrument = instrument*amplitude
        mix = mix + instrument
        newsources.append(instrument)
    return mix, newsources

def swapchannels(instruments):
    
    """ Intercambia los canales del estéreo en las fuentes."""
    newsources = []
    mix = 0
    for instrument in instruments:
        swapstate = np.random.randint(2)
        newinstrument = instrument.copy()
        if swapstate:
            newinstrument[:,0] = instrument[:,1]
            newinstrument[:,1] = instrument[:,0]
        newsources.append(newinstrument)
        mix = mix + newinstrument

    return mix, newsources

"""Técnicas de aumento de datos aplicadas a la separación de fuentes musicales 
propuestas en la tesis."""

def makemono(instruments):
    
    """Transforma las mezclas y pistas estereofónicas en monofónicas."""
    
    newsources = []
    mix = 0
    for instrument in instruments:
        monoinstrument = (instrument[:,0] + instrument[:,1])/2
        newinstrument = instrument.copy()
        newinstrument[:,0] = monoinstrument
        newinstrument[:,1] = monoinstrument
        newsources.append(newinstrument)
        mix = mix + newinstrument

    return mix, newsources


def repan(instruments):
    
    """Cambia el panorama de las fuentes en la mezcla."""
    
    newsources = []
    mix = 0
    for instrument in instruments:
        monoinstrument = (instrument[:,0] + instrument[:,1])
        newinstrument = instrument.copy()
        panindex = np.random.uniform(0,1)
        newinstrument[:,0] = panindex*monoinstrument
        newinstrument[:,1] = (1-panindex)*monoinstrument
        newsources.append(newinstrument)
        mix = mix + newinstrument

    return mix, newsources

def changepitchvocal(instruments):
    
    """ Realiza un cambio aleatorio de pitch de la voz entre +/- 2 tonos."""
    
    ncents = 100*np.random.randint(-4,4)
    fx = AudioEffectsChain().pitch(ncents)
    pitchvocals = fx(np.transpose(instruments[3]),sample_in=44100,
                                  sample_out = 44100, channels_out = 2)
    shx = np.shape(instruments[0])
    shy = np.shape(pitchvocals)
    pitchvocalscut = np.zeros((shx[0],2))
    if shy[1]<shx[0]:
        pitchvocalscut[0:shy[1],:] = np.transpose(pitchvocals[:,:])
    else:
        pitchvocalscut[:,:] = np.transpose(pitchvocals[:,0:shx[0]])   
    newsources = [instruments[0],instruments[1],instruments[2],pitchvocalscut]
    mix = instruments[0] + instruments[1] + instruments[2] + pitchvocalscut

    return mix,newsources

def addreverb(instruments):
    
    """Añade reverberación a las fuentes aplicando parámetros aleatorios"""
    
    reverberances = np.random.randint(0,100,size = (4,))
    hfdampings = np.random.randint(0,100,size = (4,))
    roomscales = np.random.randint(0,100,size = (4,))
    stereodepth = np.random.randint(0,100,size = (4,))
    predelays = np.random.randint(0,100,size = (4,))

    newsources = []
    mix = 0

    for i,instrument in enumerate(instruments):
        fx = AudioEffectsChain().reverb(reverberances[i],hfdampings[i],roomscales[i],stereodepth[i],predelays[i])
        newinstrument = fx(np.transpose(instrument),sample_in=44100, sample_out = 44100, channels_out = 2)
        newinstrument = np.transpose(newinstrument)
        newsources.append(newinstrument)
        mix = mix + newinstrument

    return mix,newsources


def distortbass(instruments):
    
    """ Añade distorsión y contenido de alta frecuencia mediante un filtro 
    shelving de agudos al bajo.
    """
    gains = np.random.randint(10,20)
    colours = np.random.randint(0,100)
    fc = np.random.randint(2000,5000)
    gainfilter = np.random.randint(0,10)

    newsources = []
    mix = 0

    newbass = instruments[0]

    fx = AudioEffectsChain().highshelf(gainfilter,fc).overdrive(gains,colours)
    newbass = fx(np.transpose(newbass),sample_in=44100,sample_out = 44100,
                 channels_out = 2)
    newbass = np.transpose(newbass)
    newsources = [newbass,instruments[1],instruments[2],instruments[3]]
    mix = newbass + instruments[1] + instruments[2] + instruments[3]

    return mix,newsources

def timestretch(instruments):
    
    """ Realiza time stretching con un factor aleatorio entre 0.75 y 1.5 de las
    fuentes.
    """

    factors = np.random.uniform(0.75,1.5,size = (4,))
    shx = np.shape(instruments[0])
    newsources = []
    mix = np.zeros((shx[0],shx[1]))

    for i,instrument in enumerate(instruments):
        fx = AudioEffectsChain().tempo(factors[i])
        newinstrument = np.zeros((shx[0],shx[1]))
        newinstrumentfx = fx(np.transpose(instrument),sample_in=44100,sample_out = 44100, channels_out = 2)
        newinstrumentfx = np.transpose(newinstrumentfx)
        shinst = np.shape(newinstrumentfx)
        if shinst[0]>shx[0]:
            newinstrument = newinstrumentfx[0:shx[0],:]
        else:
            shift = np.random.randint(0,shx[0]-shinst[0])
            newinstrument[shift:shift+shinst[0],:] = newinstrumentfx
        newsources.append(newinstrument)
        mix = mix + newinstrument

    return mix,newsources

def augmentdata(instruments):
    
    """Aplica de forma aleatoria una de las funciones definidas anteriormente."""

    transformations = {0:timestretch,1:distortbass,2:addreverb,3:changepitchvocal,4:repan,5:makemono,6:swapchannels,7:changeamplitudes}
    ntrans = np.random.randint(0,7)
    x,y = transformations[ntrans](instruments)

    return x,y

def normalizeaudio(audio):
    
    """Lleva al audio a un rango entre -1 y 1 (se trabaja en 16 bits)."""
    
    normalizedaudio = audio/(2**15-1)

    return normalizedaudio

def generateaugmentedset(n_sources,sourcesongs,nsongs,nframes,
                         framesperaugmentation):
    print("aumentando")
    """ Genera audios de mezcla nuevos con sus respectivas pistas y las guarda
    en archivos .wav de nframes muestras. Para generarlos aplica transformaciones
    a bloques de framesperaugmentation muestras y concatena los resultados. 
    A su vez, aleatoriamente mezcla pistas de distintas fuentes."""
    
    naugmentations = (nframes//framesperaugmentation)-1
    mixture = np.zeros((nframes,2))
    sources = np.zeros((nframes,2,4))
    for i in range(naugmentations):
        swap = np.random.randint(0,2)
        yraws = []
        if swap:
            songindexs = np.random.randint(0,nsongs,size=(4,))
            for k in range(n_sources):
                lengthsong = np.size(sourcesongs[songindexs[k]],1)
                sampleindex = np.random.randint(lengthsong-framesperaugmentation)
                instrumenti = sourcesongs[songindexs[k]][k,sampleindex:sampleindex+framesperaugmentation,:]
                instrumenti = normalizeaudio(instrumenti)
                yraws.append(instrumenti)
        else:
            songindex = np.random.randint(0,nsongs)
            for k in range(n_sources):
                lengthsong = np.size(sourcesongs[songindex],1)
                sampleindex = np.random.randint(lengthsong-framesperaugmentation)
                instrumenti = sourcesongs[songindex][k,sampleindex:sampleindex+framesperaugmentation,:]
                instrumenti = normalizeaudio(instrumenti)
                yraws.append(instrumenti)

        mix, newsources = augmentdata(yraws)     
        mixture[i*framesperaugmentation:(i+1)*framesperaugmentation,:] = mix[:nframes,:]
        newsources = np.transpose(np.array(newsources),(1,2,0))
        sources[i*framesperaugmentation:(i+1)*framesperaugmentation,:,:] = newsources[:nframes,:,:]        
        
    mixture = mixture.astype('float32')
    sources = sources.astype('float32')
    wavfile.write('augmented.wav',44100,mixture)
    for n, instrument in enumerate(["bass.wav","drums.wav","other.wav","vocals.wav"]):
        wavfile.write(instrument,44100,sources[:,:,n])

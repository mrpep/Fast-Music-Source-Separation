# =============================================================================
# BatchGenerator.py - Leonardo Pepino (Universidad Nacional de Tres de Febrero)
#
# This script defines a Keras generator, which manages the dataset reading and
# creates the data batches.
# =============================================================================

import numpy as np
import os
import scipy.io.wavfile as wavfile
from scipy import signal
import keras
import augmentdata

class DataGenerator(keras.utils.Sequence):
    """Generador de lotes, el cual realiza rutinas de lectura de audio en bloques,
    para manejar bases de datos de tamaño arbitrario, y realiza data augmentation
    sobre la marcha (en este caso un 50%). A su vez da el formato adecuado a 
    las entradas de la red neuronal, y calcula la STFT sobre los audios."""
	
	
    def __init__(self,batch_size=32):
        
        #Parámetros de STFT:
        self.WinType = 'hanning'
        self.WinSize = 2048
        self.HopSize = 512
        self.Overlap = self.WinSize - self.HopSize
        
        #Variables del Dataset:
        self.N_Songs = 50
        self.DatasetPath = "C:\\Datasets\\DSD100\\DSD100\\"
        self.MixturesDevPath = "Mixtures\\Dev\\"
        self.SourcesDevPath = "Sources\\Dev\\"
        self.mixturefiles = os.listdir(self.DatasetPath + self.MixturesDevPath)
        self.songorder = np.random.permutation(self.mixturefiles)
        self.sourcefilenames = ["\\bass.wav","\\drums.wav","\\other.wav","\\vocals.wav"]
        self.N_Sources = 4
        self.datasetlength = 574524609 #Numero total de samples en el dataset
        
        #Ventana Contextual:       
        self.N_FramesPast = 10
        self.N_FramesFuture = 10
        
        #Manejo de la lectura de archivos:
        self.ChunkSize = 10 #Número de canciones a cargar en RAM por vez
        self.NChunks = self.N_Songs//self.ChunkSize #Número de cargadas necesarias para cubrir dataset               
        self.Samplesize = self.HopSize*(self.N_FramesPast + self.N_FramesFuture) #Tamaño en samples de cada bloque que toma de entrada la red
        self.chunk = -1
        self.index = 0
        self.BatchSize = batch_size        
        self.epoch_i = 0
        self.mixtures = []
        self.sources = []
        self.batchsperchunk = 0 #Lleva registro de a cuanto tiene que llegar el indice para tener que leer otro bloque de canciones.
        self.fs = 44100
        self.idxaug = 0

    def on_epoch_end(self):
	
        #Al terminar la época de entrenamiento, Keras llama a este método y se reordenan las canciones y lee el primer chunk.
		
        self.songorder = np.random.permutation(self.mixturefiles)
        self.epoch_i = self.epoch_i + 1
        self.chunk = 0
        self.batchsperchunk = 0
        self.index = 0
        self.read_chunk()
              
    def read_chunk(self):
	
        #Carga un bloque de canciones en la RAM.
		
        self.mixtures = []
        self.sources = []
        self.augmentedsources = []
        chunklength = 0
        for songfilename in self.songorder[self.chunk*self.ChunkSize:(self.chunk+1)*self.ChunkSize]:
            [fs,mixturei] = wavfile.read(self.DatasetPath + self.MixturesDevPath + songfilename + "\\mixture.wav")
            chunklength = chunklength + np.size(mixturei,0)
            self.mixtures.append(mixturei)
            songsources = []
            for instrument in self.sourcefilenames:
                [fs,sourcei] = wavfile.read(self.DatasetPath + self.SourcesDevPath + songfilename + instrument)
                songsources.append(sourcei)
            sourcesi = np.array(songsources)
            self.sources.append(sourcesi)
        
        self.batchsperchunk = self.batchsperchunk + 2*chunklength//(self.Samplesize*self.BatchSize)  
        print('Augmenting data')
        augmentdata.generateaugmentedset(4,self.sources,self.ChunkSize,chunklength,100*self.Samplesize)
        
        [fs,augmentedmix] = wavfile.read("augmented.wav")
        self.augmentedmix = augmentedmix
        instruments = ['bass.wav','drums.wav','other.wav','vocals.wav']
        augmentedsources = []
        for instrument in instruments:
            [fs,sourcei] = wavfile.read(instrument)
            augmentedsources.append(sourcei)
        
        self.augmentedsources = np.array(augmentedsources)
        print(self.batchsperchunk)
        self.fs = fs    
        
    def representaudio(self,audio):
	
        #Calcula los espectrogramas
        laudio = audio[:,0]/(2**15-1)
        raudio = audio[:,1]/(2**15-1)       
        #boundary permite agregar ceros al principio y final para evitar perder esos datos con el ventaneo.
        [f,t,stftl] = signal.stft(laudio,self.fs,window = self.WinType,nperseg = self.WinSize, noverlap = self.Overlap,boundary = 'zeros')
        [f,t,stftr] = signal.stft(raudio,self.fs,window = self.WinType,nperseg = self.WinSize, noverlap = self.Overlap,boundary = 'zeros')       
        magstft = [np.log2(1+np.abs(stftl)),np.log2(1+np.abs(stftr))]

        return magstft   
    
    def __getitem__(self,idx):
	
        #Keras llama este método para obtener cada lote.
        batchx = []
        batchy = []
        self.index = self.index + 1
        if self.index > self.batchsperchunk and self.chunk < self.NChunks-1:
            self.chunk = self.chunk + 1
            self.read_chunk()
            self.idxaug = 0
        for i in range(self.BatchSize):
            da = np.random.randint(0,2)
            if da == 0:
                songindex = np.random.randint(0,self.ChunkSize)
                lengthsong = np.size(self.mixtures[songindex],0)
                sampleindex = np.random.randint(lengthsong-self.Samplesize)
                audioin = self.mixtures[songindex][sampleindex:sampleindex+self.Samplesize]
                magstft = self.representaudio(audioin)
                batchx.append(magstft)
                y = []
                for i in range(self.N_Sources):
                    instrumenti = self.sources[songindex][i,sampleindex:sampleindex+self.Samplesize,:]
                    magstfti = self.representaudio(instrumenti)
                    y.append(magstfti)
                y = np.array(y)
                batchy.append(y)
            else:
                if (self.idxaug+1)*self.Samplesize<np.size(self.augmentedmix,0):
                    audioin = self.augmentedmix[self.idxaug*self.Samplesize:(self.idxaug+1)*self.Samplesize]
                    magstft = self.representaudio(audioin)
                    batchx.append(magstft)
                    y = []
                    for i in range(self.N_Sources):
                        instrumenti = self.augmentedsources[i,self.idxaug*self.Samplesize:(self.idxaug+1)*self.Samplesize,:]
                        magstfti = self.representaudio(instrumenti)
                        y.append(magstfti)                    
                    batchy.append(y)
                    self.idxaug = self.idxaug + 1
                else:
                    self.idxaug = 0
                       
        batchx = np.array(batchx)
        batchy = np.array(batchy)
        batchx = np.transpose(batchx,(0,2,3,1))
        batchy = np.transpose(batchy,(0,3,4,2,1))
        
        return batchx, batchy
    
    def __len__(self):
		
		#Keras llama a este método para conocer el número de lotes por época.
        return int(2*self.datasetlength//(self.Samplesize*self.BatchSize))

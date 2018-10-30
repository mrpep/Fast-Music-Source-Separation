"""
BatchGenerator.py - Leonardo Pepino - 2018
"""


import numpy as np
import os
import scipy.io.wavfile as wavfile
from scipy import signal
import keras

class ValidationDataGenerator(keras.utils.Sequence):
    
    def __init__(self,batch_size=32):
        
        #Representation parameters:
        self.WinType = 'hanning'
        self.WinSize = 2048
        self.HopSize = 512
        self.Overlap = self.WinSize - self.HopSize
        
        #Dataset variables:
        self.N_Songs = 50
        self.DatasetPath = "C:\\Datasets\\DSD100\\DSD100\\"
        self.MixturesDevPath = "Mixtures\\Test\\"
        self.SourcesDevPath = "Sources\\Test\\"
        self.mixturefiles = os.listdir(self.DatasetPath + self.MixturesDevPath)
        #self.songorder = np.random.permutation(self.mixturefiles)
        self.songorder = self.mixturefiles
        self.sourcefilenames = ["\\bass.wav","\\drums.wav","\\other.wav","\\vocals.wav"]
        self.N_Sources = 4
        self.datasetlength = 574524609 #Numero total de samples en el dataset
        
        #Temporal context:       
        self.N_FramesPast = 10
        self.N_FramesFuture = 10
        
        #File management:
        self.ChunkSize = 10 #Número de canciones a cargar en RAM por vez
        self.NChunks = self.N_Songs//self.ChunkSize #Número de cargadas necesarias para cubrir dataset               
        self.Samplesize = self.HopSize*(self.N_FramesPast + self.N_FramesFuture) #Tamaño en samples de cada bloque que toma de entrada la red
        self.chunk = -1
        self.index = 0
        
        self.BatchSize = batch_size        
        self.epoch_i = 0
        #self.on_epoch_end()
        
        self.mixtures = []
        self.sources = []
        self.batchsperchunk = 0 #Lleva registro de a cuanto tiene que llegar el indice para tener que leer otro bloque de canciones.
        self.fs = 44100
        
        

    def on_epoch_end(self):
        #Reordena las canciones y lee el primer chunk
        #self.songorder = np.random.permutation(self.mixturefiles)
        self.epoch_i = self.epoch_i + 1
        self.chunk = 0
        self.batchsperchunk = 0
        self.index = 0
        self.read_chunk()
              
    def read_chunk(self):
        #Carga un bloque de canciones en la RAM.
        self.mixtures = []
        self.sources = []
        chunklength = 0
        print(self.chunk)
        for songfilename in self.songorder[self.chunk*self.ChunkSize:(self.chunk+1)*self.ChunkSize]:
            [fs,mixturei] = wavfile.read(self.DatasetPath + self.MixturesDevPath + songfilename + "\\mixture.wav")
           # mixturei = mixturei/(2**15-1)
            chunklength = chunklength + np.size(mixturei,0)
            self.mixtures.append(mixturei)
            songsources = []
            for instrument in self.sourcefilenames:
                [fs,sourcei] = wavfile.read(self.DatasetPath + self.SourcesDevPath + songfilename + instrument)
              #  sourcei = sourcei/(2**15-1)
                songsources.append(sourcei)
            sourcesi = np.array(songsources)
            self.sources.append(sourcesi)
        self.batchsperchunk = self.batchsperchunk + chunklength//(self.Samplesize*self.BatchSize)    
        print(self.batchsperchunk)
        self.fs = fs    
        
    def representaudio(self,audio):
        #Calcula los espectrogramas
        
        laudio = audio[:,0]/(2**15-1)
        raudio = audio[:,1]/(2**15-1)
        
        #boundary permite paddear principio y final para evitar perder esa data con el ventaneo.
        [f,t,stftl] = signal.stft(laudio,self.fs,window = self.WinType,nperseg = self.WinSize, noverlap = self.Overlap,boundary = 'zeros')
        [f,t,stftr] = signal.stft(raudio,self.fs,window = self.WinType,nperseg = self.WinSize, noverlap = self.Overlap,boundary = 'zeros')
        
        magstft = [np.log2(1+np.abs(stftl)),np.log2(1+np.abs(stftr))]
        #magstft = [(np.abs(stftl)),(np.abs(stftr))]
        
        return magstft   
    
    def __getitem__(self,idx):
        #Se llama para crear cada batch
        batchx = []
        batchy = []
        self.index = self.index + 1
        if self.index > self.batchsperchunk and self.chunk < self.NChunks-1:
            self.chunk = self.chunk + 1
            self.read_chunk()           
        for i in range(self.BatchSize):
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
                       
        batchx = np.array(batchx)
        batchy = np.array(batchy)
        batchx = np.transpose(batchx,(0,2,3,1))
        batchy = np.transpose(batchy,(0,3,4,2,1))
        
        return batchx, batchy
    
    def __len__(self):
        #Número de batches por epoch
        #return int(self.datasetlength//(self.Samplesize*self.BatchSize))
        return 400
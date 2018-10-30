# =============================================================================
# separate.py - Leonardo Pepino (Universidad Nacional de Tres de Febrero)
# This script allows to separate the bass, drums, other and vocals track from a
# stereo mixture. Source estimates are predicted using a pretrained convolutional
# neural network.
#
# When running the script, the user will be asked to open an audio file, and
# within seconds or minutes, depending on the availability of a GPU, estimates
# will be saved in the folder where this script is contained.
# =============================================================================

import leeraudio
import tkinter as tk
from tkinter import filedialog
import ModeloDoble
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import numpy as np

#Ventana para abrir archivos de audio:
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
#Se lee el archivo seleccionado:       
[fs,audiomixture] = leeraudio.ReadAudio(file_path)
#Se compila el modelo de Keras y se cargan los pesos:
model = ModeloDoble.CompileModel()
model.load_weights('pesos.hdf5')

eps = np.finfo(float).eps   
print("Analizando la señal")

#Se calcula la STFT, aplica el log2 de la magnitud y acondiciona el formato del tensor de entrada a la red:
[f,t,mixturestftl] = signal.stft(audiomixture[:,0]/(2**15-1),fs,window = 'hann',nperseg = 2048, noverlap = 2048-512)        
[f,t,mixturestftr] = signal.stft(audiomixture[:,1]/(2**15-1),fs,window = 'hann',nperseg = 2048, noverlap = 2048-512)     
magnitudestftin = np.array([np.log2(1+np.abs(mixturestftl)),np.log2(np.abs(mixturestftr)+1)])
sizestft = np.shape(magnitudestftin)
nframes = sizestft[2]               
magnitudestftin = np.transpose(magnitudestftin,(1,2,0))
magnitudestftin = np.reshape(magnitudestftin,(1,1025,nframes,2))

#La red neuronal genera las salidas en un bucle:
print("Realizando la separación")
stftpredicted = np.empty((1025,nframes,2,4))
for i in np.arange(10,nframes-11,3):
    print("\r" + str(np.round(i/(nframes-11)*100,decimals = 1)) + "%",end='')
    prediction = model.predict(magnitudestftin[:,:,i-10:i+11,:])   
    #Se promedian las salidas:
    stftpredicted[:,i-10:i+11,:,:] = stftpredicted[:,i-10:i+11,:,:] + (1/7)*prediction[0,:,:,:,:] 

#Se vuelven a generar máscaras suaves y aplican en la STFT original:    
stftpredicted = 2**stftpredicted - 1
denmask = np.sum(stftpredicted,axis = 3)
softmasks = np.zeros(np.shape(stftpredicted))
sources = np.zeros(np.shape(stftpredicted))
for i in range(4):
    softmasks[:,:,:,i] = np.divide(stftpredicted[:,:,:,i],denmask+eps)    
    sources[:,:,:,i] = np.multiply(softmasks[:,:,:,i],np.transpose(np.array([np.abs(mixturestftl),np.abs(mixturestftr)]),(1,2,0)))
        
magnitudestftoutl = sources[:,:,0,:]
magnitudestftoutr = sources[:,:,1,:]
        
#Fases mezcla
phasestftl = np.angle(mixturestftl)
phasestftr = np.angle(mixturestftr)        
j = np.complex(0,1)

#Se invierte la STFT de cada fuente y se guardan los resultados en archivos .wav
filename = file_path.split('/')
filename = filename[-1]
filename = filename.split('.')
filename = filename[0]
print("\nRealizando la inversión de la STFT")        
for n, instrument in enumerate([filename+"_Bass.wav",filename+"_Drums.wav",filename+"_Other.wav",filename+"_Vocals.wav"]):  
     
    [t,xl] = signal.istft((np.multiply(magnitudestftoutl[:,:,n],np.exp(phasestftl*j))), fs = fs, window = 'hann', nperseg = 2048,noverlap = 2048-512,time_axis = 1, freq_axis = 0)
    [t,xr] = signal.istft((np.multiply(magnitudestftoutr[:,:,n],np.exp(phasestftr*j))), fs = fs, window = 'hann', nperseg = 2048,noverlap = 2048-512,time_axis = 1, freq_axis = 0)       
    x = np.array([xl,xr])
    x = x.astype('float32')       
    shx = np.shape(x)
    tframes = shx[1]
    x = np.reshape(x,(2,tframes))
    x = np.transpose(x)
    x = x
    wavfile.write(instrument,fs,x)
    
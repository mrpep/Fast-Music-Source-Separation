# =============================================================================
# MisCallbacks.py - Leonardo Pepino (Universidad Nacional de Tres de Febrero)
#
# This script defines the custom loss function. Also callbacks for model reading/
# writing are supplied.
# =============================================================================

import keras
import keras.backend as k
import pickle
import tensorflow as tf

def ConfigurarTF():

    """Configura Tensorflow para un uso eficiente de la memoria de la GPU."""
	
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    k.tensorflow_backend.set_session(tf.Session(config = config))
    
def LeerModelo(modelo,weightfile,optimizerfile = None):

    """Permite cargar los parámetros de la red neuronal para continuar el 
    entrenamiento o realizar predicciones.
    Argumentos:
    modelo: es el modelo compilado de Keras al cual se le cargaran los parámetros.
    weightfile: es el archivo .hdf5 que contiene los pesos sinápticos.
    optimizerfile: es el archivo .pkl que contiene los momentos del optimizador.
    Es necesario si se quiere continuar el entrenamiento, pero no lo es para 
    realizar predicciones.
    """

    modelo.load_weights(weightfile)
    modelo._make_train_function()
    
    if optimizerfile != None:
        
        with open(optimizerfile,'rb') as f:
            weight_values = pickle.load(f)
        modelo.optimizer.set_weights(weight_values)
    
    return modelo


class GuardarModelo(keras.callbacks.Callback):

    """Clase que permite guardar al finalizar cada época de entrenamiento los 
    pesos sinápticos de la red en un archivo .hdf5 y el estado del optimizador 
    en un .pkl."""
	
    def __init__(self,filepath):
        
        self.filepath = filepath
        
    def on_epoch_end(self, epoch, logs = None):
        
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        #Guardo los pesos sinápticos:
        self.model.save_weights(filepath,overwrite = True)
        #Guardo parámetros del optimizador (necesarios si quiero reanudar entrenamiento)
        symbolic_weights = getattr(self.model.optimizer,'weights')
        weight_values = k.batch_get_value(symbolic_weights)
        with open('optimizador.pkl','wb') as f:
            pickle.dump(weight_values,f)


def CustomLossFunction(yTrue,yPred):

    """Función de costo propuesta. Toma como argumentos el espectrograma 
    verdadero y el predicho, y devuelve el error."""  
	
    BassTrue = yTrue[:,:,:,:,0]
    BassPred = yPred[:,:,:,:,0]              
    DrumsTrue = yTrue[:,:,:,:,1]
    DrumsPred = yPred[:,:,:,:,1]
    OthersTrue = yTrue[:,:,:,:,2]
    OthersPred = yPred[:,:,:,:,2]
    VocalsTrue = yTrue[:,:,:,:,3]
    VocalsPred = yPred[:,:,:,:,3]
   
    basemse = k.mean(k.square(BassPred-BassTrue))+k.mean(k.square(DrumsPred-DrumsTrue))+k.mean(k.square(VocalsPred-VocalsTrue))+0.5*k.mean(k.square(OthersPred-OthersTrue))
    diffmse = k.mean(k.square(BassPred-DrumsPred)+k.square(BassPred-OthersPred)+k.square(BassPred-VocalsPred)+
                     k.square(DrumsPred-VocalsPred)+k.square(DrumsPred-OthersPred)+k.square(VocalsPred-OthersPred),axis=-1)
    othvocmse = k.mean(k.square(VocalsPred-OthersTrue),axis=-1)
    othersmse = k.mean(k.square(VocalsPred-OthersTrue) + k.square(DrumsPred-OthersTrue) + k.square(BassPred-OthersTrue) ,axis=-1)
    recons = k.mean(k.square(BassTrue + DrumsTrue + OthersTrue + VocalsTrue - BassPred - DrumsPred - OthersPred - VocalsPred),axis=-1)
    
    alpha = 0.001
    beta = 0.01
    betav = 0.03
    err = basemse - alpha*diffmse - beta*othersmse - betav*othvocmse + 0.01*recons
    
    return err

#Métricas de performance:
    
def MetricBaseLoss(yTrue,yPred):
    BassTrue = yTrue[:,:,:,:,0]
    BassPred = yPred[:,:,:,:,0]              
    DrumsTrue = yTrue[:,:,:,:,1]
    DrumsPred = yPred[:,:,:,:,1]
    VocalsTrue = yTrue[:,:,:,:,3]
    VocalsPred = yPred[:,:,:,:,3]
   
    return k.mean(k.square(BassPred-BassTrue)+k.square(DrumsPred-DrumsTrue)+k.square(VocalsPred-VocalsTrue),axis=-1)

def MetricInterference(yTrue,yPred):
    BassPred = yPred[:,:,:,:,0]              
    DrumsPred = yPred[:,:,:,:,1]
    VocalsPred = yPred[:,:,:,:,3]
    OthersPred = yPred[:,:,:,:,2]
    return k.mean(k.square(BassPred-DrumsPred)+k.square(BassPred-OthersPred)+k.square(BassPred-VocalsPred)+
                     k.square(DrumsPred-VocalsPred)+k.square(DrumsPred-OthersPred)+k.square(VocalsPred-OthersPred),axis=-1)
    
def MetricOthVoc(yTrue,yPred):
    VocalsPred = yPred[:,:,:,:,3]
    OthersTrue = yTrue[:,:,:,:,2]
    return k.mean(k.square(VocalsPred-OthersTrue),axis=-1)

def MetricOthers(yTrue,yPred):
    BassPred = yPred[:,:,:,:,0]              
    DrumsPred = yPred[:,:,:,:,1]
    VocalsPred = yPred[:,:,:,:,3]
    OthersTrue = yTrue[:,:,:,:,2]
    
    return k.mean(k.square(VocalsPred-OthersTrue) + k.square(DrumsPred-OthersTrue) + k.square(BassPred-OthersTrue) ,axis=-1)
    
def MetricRecons(yTrue,yPred):
    BassTrue = yTrue[:,:,:,:,0]
    BassPred = yPred[:,:,:,:,0]              
    DrumsTrue = yTrue[:,:,:,:,1]
    DrumsPred = yPred[:,:,:,:,1]
    OthersTrue = yTrue[:,:,:,:,2]
    OthersPred = yPred[:,:,:,:,2]
    VocalsTrue = yTrue[:,:,:,:,3]
    VocalsPred = yPred[:,:,:,:,3]
    
    return k.mean(k.square(BassTrue + DrumsTrue + OthersTrue + VocalsTrue - BassPred - DrumsPred - OthersPred - VocalsPred),axis=-1)
    
def BassError(yTrue,yPred):
    BassTrue = yTrue[:,:,:,:,0]
    BassPred = yPred[:,:,:,:,0] 
    
    return k.mean(k.square(BassTrue-BassPred))

def VocalsError(yTrue,yPred):
    VocalsTrue = yTrue[:,:,:,:,3]
    VocalsPred = yPred[:,:,:,:,3] 
    
    return k.mean(k.square(VocalsTrue-VocalsPred))

def OthersError(yTrue,yPred):
    OthersTrue = yTrue[:,:,:,:,2]
    OthersPred = yPred[:,:,:,:,2] 
    
    return k.mean(k.square(OthersTrue-OthersPred))

def DrumsError(yTrue,yPred):
    DrumsTrue = yTrue[:,:,:,:,1]
    DrumsPred = yPred[:,:,:,:,1]
    
    return k.mean(k.square(DrumsTrue-DrumsPred))
# =============================================================================
# MisCapas.py - Leonardo Pepino (Universidad Nacional de Tres de Febrero)
#
# This script defines custom layers such as soft masks.
# =============================================================================

from keras import backend as k
from keras.layers import  Multiply, Add
from keras.layers.core import Lambda
import numpy as np
import tensorflow as tf

def divide_layer(inputs):

    """Toma como argumento una lista de 2 entradas (inputs), y devuelve la 
    división elemento a elemento de ambas."""
    	
    eps = np.finfo(float).eps   
    out = tf.div(abs(inputs[0])+eps,abs(inputs[1])+eps)
    return out
   
def softmask(layermultiout,layertomask):

    """Función para calcular máscaras suaves y aplicarlas.
    Argumentos: 
    layermultiout: lista de capas con las cuales se calcula la máscara suave a partir de sus salidas.
    layertomask: lista de capas en las que se aplica la máscara suave en la salida.
    """
    	
    bass = layermultiout[0]
    drums = layermultiout[1]
    others = layermultiout[2]
    vocals = layermultiout[3]
    
    mixture = Add()([bass,drums,others,vocals])
    
    vocbranch = Lambda(divide_layer)([vocals,mixture])
    drumsbranch = Lambda(divide_layer)([drums,mixture])
    othersbranch = Lambda(divide_layer)([others,mixture])
    bassbranch = Lambda(divide_layer)([bass,mixture])
    
    vocbranch = Multiply()([layertomask,vocbranch])
    drumsbranch = Multiply()([layertomask,drumsbranch])
    othersbranch = Multiply()([layertomask,othersbranch])
    bassbranch = Multiply()([layertomask,bassbranch])
    
    outlist = [bassbranch,drumsbranch,othersbranch,vocbranch]
    return outlist

def stacklayers(inputs):
    
    """Capa cuya salida consiste de la salida de las capas en inputs concatenadas."""    
    
    return k.stack([inputs[0],inputs[1],inputs[2],inputs[3]],axis = 4)

def unstacklayers(inputs):
    
    """Revierte el proceso de la función stacklayers"""
	
    return [inputs[:,:,:,:,0],inputs[:,:,:,:,1],inputs[:,:,:,:,2],inputs[:,:,:,:,3]]

def log2emphasis(inputs):
    
    """Capa que aplica el logaritmo en base 2 de la entrada + 1"""
	
    log2value = np.log2(2)
    emphasized = Lambda(lambda x: k.log(x+1)/log2value)(inputs)
    
    return emphasized
    
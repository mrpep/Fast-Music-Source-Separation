# =============================================================================
# ModeloDoble.py - Leonardo Pepino (Universidad Nacional de Tres de Febrero)
#
# This script compiles the keras model of the convolutional neural network
# developed. 
# =============================================================================

from MisCallbacks import CustomLossFunction, VocalsError,DrumsError,BassError,OthersError,MetricInterference,MetricOthVoc,MetricOthers,MetricRecons
from MisCapas import softmask, stacklayers, unstacklayers, log2emphasis
from keras.layers import Input, Add, BatchNormalization, Concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Reshape, Dense, Flatten, Lambda
from keras.models import Model
from keras.optimizers import Adam

def CompileModel():
    
    """Función que compila el modelo de red neuronal implementado en Keras."""
    
    #Hiperparámetros de la subred percusiva:    
    NVFiltPerc = 64
    NHFiltPerc = 64
    NReshapePerc = 1024

    #Encoder percusivo:
    stft_input = Input(shape = (1025,21,2,), dtype = 'float32')
    pvconv = Conv2D(NVFiltPerc,(1025,1),activation = "relu",use_bias = False)
    phconv = Conv2D(NHFiltPerc,(1,6),activation = "relu",use_bias = False)
    pflatter = Flatten()      
    pencoder = BatchNormalization()(stft_input)
    pencoder = pvconv(pencoder)
    pencoder = BatchNormalization()(pencoder)
    pencoder = phconv(pencoder)
    pencoder = BatchNormalization()(pencoder)
    pencoder = pflatter(pencoder)
    
    #Hiperparámetros de la subred armónica:
    NVFiltHarm = 64
    NHFiltHarm = 32
    NReshapeHarm = 1536
    
    #Encoder armónico:
    
    hhconv = Conv2D(NHFiltHarm,(1,21),activation = "relu",use_bias = False)
    hvconv = Conv2D(NVFiltHarm,(82,1),activation = "relu",use_bias = False,strides = (41,1))    
    hflatter = Flatten()       
    hencoder = BatchNormalization()(stft_input)
    hencoder = hhconv(hencoder)
    hencoder = BatchNormalization()(hencoder)
    hencoder = hvconv(hencoder)
    hencoder = BatchNormalization()(hencoder)
    hencoder = hflatter(hencoder)
    
    #Espacio Latente:
    latentspace = Concatenate()([hencoder,pencoder])
    latentspace = Dense(1024,activation = "relu",use_bias = False)(latentspace)

    #Capas de convolución transpuesta con pesos atados entre si (decoder percusivo):
    psharedHDeconv2D = Conv2DTranspose(NVFiltPerc,(1,6),activation = "relu")
    psharedVDeconv2D = Conv2DTranspose(2,(1025,1),activation = "relu")
    
    #Decodificadores paralelos percusivos para cada instrumento:
    pbassbranch = Dense(NReshapePerc,activation = "relu")(latentspace)
    pbassbranch = Reshape((1,16,NHFiltPerc))(pbassbranch)
    pbassbranch = psharedHDeconv2D(pbassbranch)
    pbassbranch = psharedVDeconv2D(pbassbranch)
    
    pdrumsbranch = Dense(NReshapePerc,activation = "relu")(latentspace)
    pdrumsbranch = Reshape((1,16,NHFiltPerc))(pdrumsbranch)
    pdrumsbranch = psharedHDeconv2D(pdrumsbranch)
    pdrumsbranch = psharedVDeconv2D(pdrumsbranch)
    
    pothersbranch = Dense(NReshapePerc,activation = "relu")(latentspace)
    pothersbranch = Reshape((1,16,NHFiltPerc))(pothersbranch)
    pothersbranch = psharedHDeconv2D(pothersbranch)
    pothersbranch = psharedVDeconv2D(pothersbranch)
    
    pvocbranch = Dense(NReshapePerc,activation = "relu")(latentspace)    
    pvocbranch = Reshape((1,16,NHFiltPerc))(pvocbranch)
    pvocbranch = psharedHDeconv2D(pvocbranch)
    pvocbranch = psharedVDeconv2D(pvocbranch)
    
    #Salidas de los decodificadores paralelos de la subred percusiva:
    
    pbass = psharedVDeconv2D.get_output_at(0)
    pdrums = psharedVDeconv2D.get_output_at(1)
    pothers = psharedVDeconv2D.get_output_at(2)
    pvocals = psharedVDeconv2D.get_output_at(3)

    poutput = Lambda(stacklayers)([pbass,pdrums,pothers,pvocals])

    #Capas de convolución transpuesta con pesos atados entre si (decoder armónico):
    hsharedVDeconv2D = Conv2DTranspose(32,(82,1),activation = "relu",strides = (41,1))
    hsharedHDeconv2D = Conv2DTranspose(2,(1,21),activation = "relu")
    
    #Decodificadores paralelos armónicos para cada instrumento:
    hbassbranch = Dense(NReshapeHarm,activation = "relu")(latentspace)
    hbassbranch = Reshape((24,1,NVFiltHarm))(hbassbranch)
    hbassbranch = hsharedVDeconv2D(hbassbranch)
    hbassbranch = hsharedHDeconv2D(hbassbranch)
     
    hdrumsbranch = Dense(NReshapeHarm,activation = "relu")(latentspace)
    hdrumsbranch = Reshape((24,1,NVFiltHarm))(hdrumsbranch)
    hdrumsbranch = hsharedVDeconv2D(hdrumsbranch)
    hdrumsbranch = hsharedHDeconv2D(hdrumsbranch)
        
    hothersbranch = Dense(NReshapeHarm,activation = "relu")(latentspace)
    hothersbranch = Reshape((24,1,NVFiltHarm))(hothersbranch)
    hothersbranch = hsharedVDeconv2D(hothersbranch)
    hothersbranch = hsharedHDeconv2D(hothersbranch)
        
    hvocbranch = Dense(NReshapeHarm,activation = "relu")(latentspace)    
    hvocbranch = Reshape((24,1,NVFiltHarm))(hvocbranch)
    hvocbranch = hsharedVDeconv2D(hvocbranch)
    hvocbranch = hsharedHDeconv2D(hvocbranch)
    
    hbass = hsharedHDeconv2D.get_output_at(0)
    hdrums = hsharedHDeconv2D.get_output_at(1)
    hothers = hsharedHDeconv2D.get_output_at(2)
    hvocals = hsharedHDeconv2D.get_output_at(3)
    
    houtput = Lambda(stacklayers)([hbass,hdrums,hothers,hvocals])
    
    #Fusión de las salidas de los decodificadores armónico y percusivo:
    houtput = Lambda(lambda x: 2**x-1)(houtput)
    poutput = Lambda(lambda x: 2**x-1)(poutput)
    totaloutput = Add()([houtput,poutput])
    
    sourceoutputs = Lambda(unstacklayers)(totaloutput)

    #Cálculo y aplicación de la máscara suave:
    salida = softmask(sourceoutputs,stft_input)
    finaloutput = Lambda(stacklayers)(salida)
    finaloutput = Lambda(log2emphasis)(finaloutput)
    
    #Definición del modelo de Keras:
    modelodoble = Model(inputs = stft_input,outputs = finaloutput)
    #Especificación del optimizador:
    opt = Adam(lr = 0.01,clipvalue = 0.9,amsgrad = True)
    #Se compila el modelo utilizando como función de pérdida la propuesta. También se especifican errores a mostrar durante el entrenamiento con el fin de monitorear el progreso.
    modelodoble.compile(loss = CustomLossFunction,optimizer = opt,metrics = [VocalsError,DrumsError,BassError,OthersError,MetricInterference,MetricOthVoc,MetricOthers,MetricRecons])
    
    return modelodoble
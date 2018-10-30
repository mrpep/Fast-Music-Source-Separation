# =============================================================================
# trainmodel.py - Leonardo Pepino (Universidad Nacional de Tres de Febrero)
#
# This script sets the training parameters. In the last line, number of epochs 
# can be modified. Also, Tensorboard setting is possible.
# =============================================================================

from BatchGenerator import DataGenerator
from ValidationGenerator import ValidationDataGenerator
from keras.callbacks import TensorBoard
from MisCallbacks import GuardarModelo

def trainmodel(model):
    """Función que configura el entrenamiento del modelo y lo ejecuta."""
    #Se usan generadores los cuales levantan los lotes de datos para entrenar la red:
    training_generator = DataGenerator()
    validation_generator = ValidationDataGenerator()

    #Guardado de pesos y estado del optimizador en cada época:
    filepath = "weights-{epoch:02d}.hdf5"
    checkpoint = GuardarModelo(filepath)
    
    #Despliegue de estadísticas de entrenamiento en Tensorboard
    tbCallBack = TensorBoard(log_dir = './Graph', histogram_freq = 0, write_graph = True,write_images = False)    
    callbacklist = [checkpoint,tbCallBack]

    #Mediante esta función se entrena la red:
    model.fit_generator(generator=training_generator,validation_data=validation_generator,
                        epochs = 20, callbacks = callbacklist)
    
    
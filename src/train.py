# =============================================================================
# train.py - Leonardo Pepino (Universidad Nacional de Tres de Febrero)
#
# This script allows to train the implemented convolutional neural network.
# It is necessary to specify dataset folder location in BatchGenerator.py and
# ValidationGenerator.py. It is possible to restart training if files with weights  
# and optimizer state are available.
# =============================================================================

import ModeloDoble
import trainmodel
from MisCallbacks import LeerModelo, ConfigurarTF

ConfigurarTF()
model = ModeloDoble.CompileModel()
#Descomentar esta linea si se pausó el entrenamiento previamente:
#model = LeerModelo(model,'weights-11.hdf5','optimizador.pkl')
trainmodel.trainmodel(model)


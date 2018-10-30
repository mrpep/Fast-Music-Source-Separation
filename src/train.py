"""Script que ejecuta el entrenamiento del modelo. Permite leer un modelo ya entrenado y continuar el entrenamiento."""

import ModeloDoble
import trainmodel
from MisCallbacks import LeerModelo, ConfigurarTF

ConfigurarTF()
model = ModeloDoble.CompileModel()
#Comentar esta linea si el entrenamiento es desde cero:
model = LeerModelo(model,'weights-11.hdf5','optimizador.pkl')
trainmodel.trainmodel(model)


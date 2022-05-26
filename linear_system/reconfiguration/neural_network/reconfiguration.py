import numpy as np
import keras
from keras.layers.normalization import batch_normalization

class Reconfiguration():
    def __init__(self, input_shape, neural_network=[]):
        if neural_network == []:
            self.model = keras.Sequential()
            self.model.add( keras.Dense(16, input_dim=8, activation='relu', input_shape=input_shape) )
            self.model.add( batch_normalization() )
            self.model.add( keras.Dense(8, input_dim=8, activation='relu') )
            self.model.add( batch_normalization() )
            self.model.add( keras.Dense(4, input_dim=8, activation='relu') )
            self.model.add( batch_normalization() )
            self.model.add( keras.Dense(2, input_dim=8, activation='relu') )
            self.model.compile(optimizer = 'adam', loss="mean_squared_error", metrics=['accuracy'])

    def train(self, datasetX, datasetY):
        self.model.fit(datasetX, datasetY, batchsize=512, validation_split=0.1, epochs=2)

    def reconfigure(self, data):
        return self.model.predict(data)

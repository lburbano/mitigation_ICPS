import numpy as np

class Reconfiguration():
    def __init__(self, a, b, Ts, input_dimension):
        self.a  = a
        self.b  = b
        self.Ts = Ts
        self.input_dimension = input_dimension
    
    def reconfigure(self, y, y_prediction, y_previous, y_prediction_previous, t):
        difference = ( y_prediction - y ) - np.exp(self.a * self.Ts) * (y_prediction_previous - y_previous)
        ua = self.a/self.b * ( difference ) / (1 - np.exp( self.a*(self.Ts) ))
        # ua = -self.a/self.b * (np.exp(self.Ts) - np.exp( self.a*(t-self.Ts) )) * ( y_prediction - y )
        return ua
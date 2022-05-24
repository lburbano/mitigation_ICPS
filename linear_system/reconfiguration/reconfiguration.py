import numpy as np

class Reconfiguration():
    def __init__(self, a, b, Ts, input_dimension):
        self.a  = a
        self.b  = b
        self.Ts = Ts
        self.input_dimension = input_dimension
    
    def reconfigure(self, y, yp, t):
        ua = self.a/self.b * ( yp - y ) / (1 - np.exp( self.a*(self.Ts) ))
        # ua = -self.a/self.b * (np.exp(self.Ts) - np.exp( self.a*(t-self.Ts) )) * ( yp - y )
        return ua
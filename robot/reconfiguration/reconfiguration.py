import numpy as np
from system.system import *
from numpy.linalg import pinv, inv
from scipy.linalg import expm

class Reconfiguration():
    def __init__(self, Ts, input_dimension):
        self.Ts = Ts
        self.input_dimension = input_dimension
    
    def reconfigure(self, u, y, y_estimation, y_previous, y_estimation_previous, t):
        y_estimation = y_estimation.flatten()
        u = u.flatten()
        y = y.flatten()
        y_previous = y_previous.flatten()
        y_estimation_previous = y_estimation_previous.flatten()
        A = jacobian_f(y_estimation, u, self.Ts)
        B = jacobian_B(y_estimation, u, self.Ts)

        difference = y_estimation - y - np.matmul(expm(A*self.Ts), y_estimation_previous - y_previous)

        # ua = np.matmul( y_estimation - y, A) 
        ua = np.matmul( difference, A) 
        exponential = inv( np.eye(len(y)) - expm(A*self.Ts) )
        ua = np.matmul( ua, exponential)
        ua = np.matmul( pinv(B), ua)
        # ua = self.a/self.b * ( y_estimation - y ) / (1 - np.exp( self.a*(self.Ts) ))
        return ua.reshape(self.input_dimension)
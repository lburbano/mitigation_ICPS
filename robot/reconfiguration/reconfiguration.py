import numpy as np
from numpy.linalg import pinv, inv
from scipy.linalg import expm

class Reconfiguration():
    def __init__(self, jacobian_A, jacobian_B, Ts, input_dimension):
        self.Ts = Ts
        self.input_dimension = input_dimension
        self.A = jacobian_A
        self.B = jacobian_B
    
    def reconfigure(self, u, y, y_estimation, y_previous, y_estimation_previous, t):
        y_estimation = y_estimation.flatten()
        u = u.flatten()
        y = y.flatten()
        y_previous = y_previous.flatten()
        y_estimation_previous = y_estimation_previous.flatten()
        A = self.A(y_estimation, u)
        B = self.B(y_estimation, u)

        difference = -np.matmul(expm(-A*self.Ts), y_estimation - y) + ( y_estimation_previous - y_previous )
        integral = self.Ts/2*( B + np.matmul( expm( -A*self.Ts ), B) )
        
        '''
        difference = y_estimation - y - np.matmul(expm(A*self.Ts), y_estimation_previous - y_previous)

        # ua = np.matmul( y_estimation - y, A) 
        ua = np.matmul( expm(-A*self.Ts), difference) 
        ua = np.matmul( A, ua)
        exponential = inv( expm(-A*self.Ts) - np.eye(len(y)) )
        ua = np.matmul( exponential, ua)
        ua = np.matmul( pinv(B), ua)
        # ua = self.a/self.b * ( y_estimation - y ) / (1 - np.exp( self.a*(self.Ts) ))
        '''
        ua = np.matmul( pinv(integral), difference)
        return ua.reshape(self.input_dimension)
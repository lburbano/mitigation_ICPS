import numpy as np
from system.system import *
from numpy.linalg import inv

class EKF():
    def __init__(self, initial_state, state_dimension, input_dimension, Q, R, P, sampling_time):
        self.Q = Q
        self.R = R
        self.P = P
        self.sampling_time = sampling_time
        self.state_dimension = state_dimension
        self.input_dimension = input_dimension
        self.xp  = np.array(initial_state).reshape(state_dimension)
        self.xpn = np.array(initial_state).reshape(state_dimension)
        self.xe  = np.array(initial_state).reshape(state_dimension)
        self.F   = np.zeros( (3,3) )
        self.H   = np.zeros( (3,3) )
        self.j_f = lambda x, u:jacobian_f(x, u, sampling_time) 
        self.j_h = lambda x, u:jacobian_h(x, u, sampling_time)
        pass

    def predict(self, u):
        u = u.flatten()
        xe = self.xp.flatten()
        self.xp = xe + system_flow(xe, u)*self.sampling_time
        self.xp = self.xp.reshape(self.state_dimension)
        return self.xp
    def estimate(self, y, u):
        self.xe = self.xe.flatten()
        self.xpn = self.xpn.flatten()
        u = u.flatten()
        y = y.flatten()
        self.F   = self.j_f(self.xpn, u)
        self.xpn = self.xe + system_flow(self.xe, u)*self.sampling_time
        self.H   = self.j_h(self.xpn, u)
        self.P   = np.matmul( self.F, np.matmul(self.P, self.F) ) + self.Q
        ye       = y - self.xpn
        S        = np.matmul( self.H, np.matmul(self.P, self.H) ) + self.R
        K        = np.matmul( self.P, np.matmul(self.H.transpose(), inv(S)) )
        self.xe  = self.xpn + np.matmul(K, ye)
        self.P   = np.matmul( np.eye(self.state_dimension[0]) - np.matmul(K, self.H), self.P )
        self.xe  = self.xe.reshape(self.state_dimension)
        self.xpn = self.xpn.reshape(self.state_dimension)
        return self.xe, self.xpn

    def recover(self, y):
        self.xp = y.reshape(self.state_dimension)
        self.xe = y.reshape(self.state_dimension)
    
    def reset_prediction(self):
        self.xpn = self.xp
    
    def reset(self, initial_state, state_dimension, Q, R, P, sampling_time):
        self.Q = Q
        self.R = R
        self.P = P
        self.sampling_time = sampling_time
        self.state_dimension = state_dimension
        self.xp  = initial_state
        self.xe  = initial_state
        self.j_f = lambda x, u:jacobian_f(x, u, sampling_time) 
        self.j_h = lambda x, u:np.eye(2)
        self.F  = np.zeros( (3,3) )
        self.H  = np.zeros( (3,3) )
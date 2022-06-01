import numpy as np
from system.system import *
from numpy.linalg import inv

class EKF():
    def __init__(self, j_f, j_h, h, dynamics, initial_state, state_dimension, input_dimension, Q, R, P, sampling_time):
        self.Q = Q
        self.R = R
        self.P = P
        self.sampling_time = sampling_time
        self.state_dimension = state_dimension
        self.input_dimension = input_dimension
        self.xp  = np.array(initial_state).reshape(state_dimension)
        self.xe  = np.array(initial_state).reshape(state_dimension)
        self.F   = np.zeros( (3,3) )
        self.H   = np.zeros( (3,3) )
        self.j_f = j_f
        self.j_h = j_h
        self.jump = dynamics
        self.h = h
        pass

    def predict(self, u):
        u = u.flatten()
        xe = self.xp.flatten()
        xp = self.jump(self.xe, u)
        xp = xp.reshape(self.state_dimension)
        return xp
    def estimate(self, y, u):
        self.xe = self.xe.flatten()
        self.xp = self.xp.flatten()
        u = u.flatten()
        y = y.flatten()
        self.F   = self.j_f(self.xp, u)
        self.xp  = self.predict(u)
        self.H   = self.j_h(self.xp, u)
        self.P   = np.matmul( self.F, np.matmul(self.P, self.F) ) + self.Q
        ye       = y - self.h(self.xp)
        S        = np.matmul( self.H, np.matmul(self.P, self.H.transpose()) ) + self.R
        K        = np.matmul( self.P, np.matmul(self.H.transpose(), inv(S)) )
        self.xe  = self.xp.flatten() + np.matmul(K, ye.flatten())
        self.P   = np.matmul( np.eye(self.state_dimension[0]) - np.matmul(K, self.H), self.P )
        self.xe  = self.xe.reshape(self.state_dimension)
        self.xp  = self.xp.reshape(self.state_dimension)
        return self.xe, self.xp

    def recover(self, y):
        self.xp = y.reshape(self.state_dimension)
        self.xe = y.reshape(self.state_dimension)
    
    def reset_prediction(self):
        self.xpn = self.xp
    
    # def reset(self, initial_state, state_dimension, Q, R, P, sampling_time):
    #     self.Q = Q
    #     self.R = R
    #     self.P = P
    #     self.sampling_time = sampling_time
    #     self.state_dimension = state_dimension
    #     self.xp  = initial_state
    #     self.xe  = initial_state
    #     self.j_f = lambda x, u:discrete_jacobian_f(x, u, sampling_time) 
    #     self.j_h = lambda x, u:np.eye(2)
    #     self.F  = np.zeros( (3,3) )
    #     self.H  = np.zeros( (3,3) )
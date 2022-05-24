import numpy as np

class Controller(object):
    def __init__(self, target, state_dimension, input_dimension, Ts):
        self.state_dimension = state_dimension
        self.input_dimension = input_dimension
        self.alarm = np.zeros(state_dimension)
        self.target = target 
        self.Ts = Ts

        self.pos_i = 0
        self.ang_i = 0
        self.pos_Kp = 0.5
        self.ang_kp = 1
        self.turned = 0
        pass

    def update_alarm(self, alarm):
        self.alarm = alarm

    def update_u(self, x):
        u = np.zeros(self.input_dimension)
        error = np.zeros(self.input_dimension)
        error[0] = self.target[0] - x[0]
        error[1] = self.target[1] - x[1]

        u[0] = ( (error[0])**2 + (error[1])**2 )**(1/2) 
        u[1] = np.arctan2( error[1], error[0] )
        # if u[0] < 0.05:
        #     u[0] = 0
        #     u[1] = 0

        # self.pos_i = self.pos_i + u[0] * self.Ts
        # self.ang_i = self.ang_i + u[1] * self.Ts

        
        # x[2] = np.mod( x[2], np.pi) + 0.0
        if np.abs( u[1] - np.mod(x[2], np.pi) ) > 0.0001 and self.turned == 0:
            u[0] = 0
            u[1] = u[1] - np.mod(x[2], np.pi)
        elif np.abs( u[0] ) > 0.01:
            self.turned = 1
            # u[0] = 1
            u[1] = 0
            # self.turned = 1
        else:
            u[0] = 0
            u[1] = 0
        
        u[0] = u[0]*self.pos_Kp
        u[1] = u[1]*self.ang_kp


        if sum(self.alarm) > 0:
            u = np.array([0, 0]).reshape( self.input_dimension )
            pass
        
        u[0] = np.clip(u[0], -1, 1)
        u[1] = np.clip(u[1], -1, 1)
        return u
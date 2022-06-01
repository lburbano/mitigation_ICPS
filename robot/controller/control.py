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

    def update_u(self, x, u_reconfiguration):
        u = np.zeros(self.input_dimension)
        x = x.flatten()

        d = 0.035
        z1 = x[0] + d*np.cos( x[2] )
        z2 = x[1] + d*np.sin( x[2] )

        error = self.target.flatten()*self.pos_Kp - np.array([z1, z2]).flatten()*self.pos_Kp


        v = 1/d * (d*np.cos(x[2]) * error[0] + d*np.sin(x[2]) * error[1] )
        w = 1/d * (-np.sin(x[2]) * error[0] + np.cos(x[2]) * error[1] )


        return np.array([v, w]).reshape(self.input_dimension)
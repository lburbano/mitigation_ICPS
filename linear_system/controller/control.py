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

    def update_u(self, x, ua):
        u = np.zeros(self.input_dimension)
        u[0] = 1 - ua
        return u
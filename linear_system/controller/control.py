import numpy as np

class Controller(object):
    def __init__(self, target, state_dimension, input_dimension, Ts):
        self.state_dimension = state_dimension
        self.input_dimension = input_dimension
        self.alarm = np.zeros(state_dimension)
        self.target = target 
        self.Ts = Ts

        self.i = 0
        self.Kp = -1
        self.Ki = 0.3
        self.turned = 0
        pass

    def update_alarm(self, alarm):
        self.alarm = alarm

    def update_u(self, x, ua):
        u = np.zeros(self.input_dimension)
        e = (self.target - x)
        self.i = self.i + self.Ts * e
        u[0] = self.Kp * x + self.Ki*self.i
        u[0] = u[0] - ua
        return u
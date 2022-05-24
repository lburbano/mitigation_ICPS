import numpy as np
from .ekf.ekf import *

class Estimator():
    def __init__(self, a, b, estimator, initial_state, state_dimension, input_dimension, sampling):
        self.xp = initial_state
        self.xe = initial_state
        self.state_dimension = state_dimension
        self.input_dimension = input_dimension
        self.sampling_time = sampling
        self.estimator_name = estimator
        self.alarm = np.zeros(state_dimension)
        self.store_y = np.zeros( (state_dimension[0], 2) )
        self.store_u = np.zeros( (input_dimension[0], 2) )
        self.recover_store = 0

        self.a = a
        self.b = b

    def store_data(self, y, u):
        self.store_y[:, 1:] = self.store_y[:, 0:-1]
        self.store_y[:, 0] = y.flatten()

        self.store_u[:, 1:] = self.store_u[:, 0:-1]
        self.store_u[:, 0] = u.flatten()



    def estimate_ekf(self, y, u):
        self.xe, self.xp = self.estimator.estimate(y, u)
        return self.xe, self.xp
    def predict_ekf(self, u):
        return self.estimator.predict(u)
    
    def perfect_estimator(self, y, u):
        self.estimator.step(u)
        t, self.xp = self.estimator.observe_without_attack()
        self.xe = self.xp
        return self.xe, self.xp

    def set_estimator(self):
        if self.estimator_name == "perfect":
            self.estimator = System(self.a, self.b, self.xp, self.state_dimension, self.input_dimension, attack=0, sampling=self.sampling_time)
        elif self.estimator_name == "ekf":
            Q = np.eye(self.state_dimension[0])
            R = np.eye(self.state_dimension[0])
            P = np.eye(self.state_dimension[0])
            self.estimator = EKF(self.a, self.b, self.xp, self.state_dimension, self.input_dimension, Q, R, P, self.sampling_time)

    def estimate(self, y, u):
        if self.estimator_name == "perfect":
            xe, xp = self.perfect_estimator(y, u)
        elif self.estimator_name == "ekf":
            xe, xp = self.estimate_ekf(y, u)
        self.store_data(xp, u)
        return xe, xp

    def recover_estimator(self):
        if self.estimator_name == "perfect":
            pass # do nothing
        elif self.estimator_name == "ekf":
            if sum(self.alarm) > 0:
                if not self.recover_store:
                    Q = np.eye(3)
                    R = np.eye(2)
                    P = np.eye(3)
                    self.estimator.recover(self.store_y[:, -1])
                    # self.estimator.reset(self.store_y[:, -1], self.state_dimension, Q, R, P, self.sampling_time)
                    self.estimator.predict(self.store_u[:, -1])
                    self.estimator.predict(self.store_u[:, 0])
                    self.recover_store = 1
                    
    
    def update_alarm(self, alarm):
        self.alarm = alarm
        self.recover_estimator()

import numpy as np
from estimator.estimator import *

class AnomalyDetector():
    def __init__(self, estimator_name, initial_state, state_dimension, input_dimension, threshold, sampling):
        self.residues = np.zeros(state_dimension)
        self.state_dimension = state_dimension
        self.alarm = np.zeros(state_dimension)
        self.threshold = threshold

        self.estimation = np.array(initial_state).reshape(state_dimension)
        self.prediction = np.array(initial_state).reshape(state_dimension)
        self.measurement = initial_state
        self.estimator   = Estimator(estimator_name, initial_state, state_dimension, input_dimension, sampling)
        self.estimator.set_estimator( )

    def set_estimator(self, Q = None, R = None, P = None, j_f = None, j_h = None, output_function=None, discrete_dynamics = None ):
        self.estimator.set_estimator(Q, R, P, j_f, j_h, output_function, discrete_dynamics)

    def compute_residues(self):
        self.residues = np.abs( self.measurement - self.prediction )
        return self.residues
    
    def verify_alarm(self):
        for i in range( self.state_dimension[0] ):
            self.alarm[i] = 0
            if self.residues[i] > self.threshold[i]:
                self.alarm[i] = 1
        self.alarm = self.alarm.reshape(self.state_dimension)
        return self.alarm
    
    def estimate(self, y, u):
        self.estimation, self.prediction = self.estimator.estimate(y, u)
        return self.estimation, self.prediction
    
    def update_measurement(self, measurements):
        self.measurement = measurements
    
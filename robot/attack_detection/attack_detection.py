import numpy as np
from system.system import *
from estimator.estimator import *

class AnomalyDetector():
    def __init__(self, estimator_name, initial_state, state_dimension, input_dimension, threshold, sampling):
        self.residues = np.zeros(state_dimension)
        self.state_dimension = state_dimension
        self.alarm = np.zeros(state_dimension)
        self.threshold = threshold
        self.system = System(initial_state, state_dimension, input_dimension, attack=0, sampling=sampling, noise=0)

        self.estimation = np.array(initial_state).reshape(state_dimension)
        self.prediction = np.array(initial_state).reshape(state_dimension)
        self.measurement = initial_state
        self.estimator   = Estimator(estimator_name, initial_state, state_dimension, input_dimension, sampling)
        self.estimator.set_estimator( )

    
    def compute_residues(self):
        self.residues = np.abs( self.measurement - self.prediction )
        return self.residues
    
    def verify_alarm(self):
        for i in range( self.state_dimension[0] ):
            self.alarm[i] = 0
            if self.residues[i] > self.threshold[i]:
                self.alarm[i] = 1
        return self.alarm
    
    def estimate(self, y, u):
        self.estimation, self.prediction = self.estimator.estimate(y, u)
        return self.estimation, self.prediction
    
    def update_measurement(self, measurements):
        self.measurement = measurements
    
    def reset(self, initial_state):
        self.system.reset(initial_state)
        self.alarm = np.zeros(self.state_dimension)
        self.residues = np.zeros(self.state_dimension)
        self.estimation = initial_state
import numpy as np
from estimator.ekf.ekf import *

class InputEsitmator():
    def __init__(self, j_f, j_h, h, dynamics, initial_state, sampling_time, state_dimension, input_dimension):
        self.input_dimension = input_dimension
        self.state_dimension = state_dimension
        self.state_dimension_augmented = ( input_dimension[0] + state_dimension[0], 1 )
        self.sampling_time = sampling_time
        initial_state = np.array(initial_state).reshape(self.state_dimension_augmented)

        j_f = j_f
        j_h = j_h
        dynamics = dynamics
        h = h

        # move this matrices to the main. This should be a parameter
        Q = np.eye(self.state_dimension_augmented[0])
        R = np.eye(self.state_dimension[0])*100
        P = np.eye(self.state_dimension_augmented[0])
        self.estiamor = EKF(j_f, j_h, h, dynamics, initial_state, self.state_dimension_augmented, input_dimension, Q, R, P, sampling_time)

    def estimate_u(self, y, u):
        xe, xp = self.estiamor.estimate(y, u)
        return xe
    
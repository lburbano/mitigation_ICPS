import numpy as np
from estimator.ekf.ekf import *
from system.system import *

def augmented_discrete_dynamics(state, u, dt, a, b):
    xp = np.zeros( len(state) )
    # xp_system = 
    pass

def augmented_jacobian_state(state, u, dt, a, b):
    pass

def augmented_jacobian_input(state, u, dt, a, b):
    pass

class InputEsitmator():
    def __init__(self, state_dimension, input_dimension) -> None:
        self.input_dimension = input_dimension
        self.state_dimension = state_dimension
        self.state_dimension_augmented = ( input_dimension[0] + state_dimension[0], 1 )

    
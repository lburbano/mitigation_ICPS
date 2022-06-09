import numpy as np
from scipy.integrate import solve_ivp
from functools import partial

def system_flow(state, u):
    flow = np.zeros( len(state) )
    flow[0] = u[0] * np.cos(state[2])
    flow[1] = u[0] * np.sin(state[2])
    flow[2] = u[1]
    return flow

def output_function(state):
    return state

def discrete_system(state, u, dt):
    x = np.zeros( len(state) )
    x[0] = state[0] + u[0] * np.sin(x[2]) * dt
    x[1] = state[1] + u[0] * np.cos(x[2]) * dt
    x[2] = state[2] + u[1] * dt
    return x

def discrete_jacobian_f(state, u, dt):
    return np.array( [[1, 0, -dt*u[0]*np.sin(state[2])],\
    [0, 1, dt*u[0]*np.cos(state[2])],\
    [0, 0, 1]] )

def discrete_jacobian_h(state, u, dt):
    return np.eye(len(state))

def jacobian_B(state, u, dt):
    return np.array( [ [np.cos(state[2]), 0],\
        [np.sin(state[2]), 0],\
        [0, 1] ] )

def jacobian_A(state, u, dt):
    return np.array( [[0, 0, -u[0]*np.sin(state[2])],\
    [0, 0, u[0]*np.cos(state[2])],\
    [0, 0, 0]] )

def augmented_discrete_system(state, u, dt):
    jumps = np.zeros( len( state ) )
    z1 = state[0]
    z2 = state[1]
    z3 = state[2]
    t1 = state[3]
    t2 = state[4]
    jumps[0] = z1 + dt*(u[0] + t1)*np.sin(z3)
    jumps[1] = z2 + dt*(u[0] + t1)*np.cos(z3)
    jumps[2] = z3 + dt*(u[1] + t2)
    jumps[3] = t1
    jumps[4] = t2
    return jumps

def augmented_discrete_jacobian_f(state, u, dt):
    n = len(state)
    z1 = state[0]
    z2 = state[1]
    z3 = state[2]
    t1 = state[3]
    t2 = state[4]

    jacobian = np.eye( n )
    jacobian[0, 2] =  dt * np.cos(z3) * (t1 + u[0]); jacobian[0, 3] = dt * np.sin(z3)
    jacobian[1, 2] = -dt * np.sin(z3) * (t1 + u[0]); jacobian[1, 3] = dt * np.cos(z3)
    jacobian[2, 4] =  dt
    return jacobian

def augmented_discrete_jacobian_h(state, u, dt):
    return np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])

def augmented_output_function(state):
    return state[0:3]

class System(object):
    def __init__(self,  initial_state, state_dimension, input_dimension, attack=0, noise=0, sampling=0.1, atol=1e-8, rtol=1e-8, max_step=0.001):
        self.x  = np.array( initial_state ).reshape( state_dimension )
        self.state_dimension = state_dimension
        self.input_dimension = input_dimension
        self.ts = sampling
        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step
        self.t = 0
        self.attack_indicator = attack
        self.F = lambda x,u:system_flow(x, u)
        self.noise = noise
        
    
    def step(self, u):
        dyna = partial(self._flow, u = u)
        sol = solve_ivp(dyna, [self.t, self.t + self.ts], self.x[:,-1].flatten(), max_step=self.max_step, atol=self.atol, rtol=self.rtol, method='RK23', dense_output=True)

        t = sol.t
        x = sol.y
        self.t = t[-1]
        self.x = np.array(x[:, -1]).reshape(self.state_dimension)
        
        return t, x
    
    def observe_with_attack(self):
        if self.t > 1 and self.t < 15 and self.attack_indicator:
            measurement = self.x + np.array( [0, 1, 0] ).reshape( self.state_dimension )
        else:
            measurement = self.x + np.array( [0, 0, 0] ).reshape( self.state_dimension )
        return self.t, measurement
    
    
    def observe_without_attack(self):
        return self.t, self.x

    def _flow(self, t, state, u):
        # beta = self.lr/(self.lf + self.lr) * np.tan( u[1] )
        mean = [0, 0, 0]
        cov = np.eye(self.state_dimension[0])
        flow = np.zeros( len(state) )
        flow = self.F(state, u) 
        return flow

    
    def reset(self, initial_state):
        self.x = np.array( initial_state ).reshape(self.state_dimension)
        self.t = 0
        return self.t, self.x
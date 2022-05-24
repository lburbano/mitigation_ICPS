import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
'''
def system_flow(state, u):
    flow = np.zeros( len(state) )
    flow[0] = u[0] * np.cos(state[2])
    flow[1] = u[0] * np.sin(state[2])
    flow[2] = u[1]
    return flow

def jacobian_f(state, u, dt):
    return np.array( [[1, 0, -dt*u[0]*np.sin(state[2])],\
    [0, 1, dt*u[0]*np.cos(state[2])],\
    [0, 0, 1]] )

def jacobian_h(state, u, dt):
    return np.eye(len(state))
'''
def system_flow(state, u, a, b):
    flow = np.zeros( len(state) )
    flow[0] = a*state + b*u
    return flow

def jacobian_f(state, u, dt, a, b):
    return np.array( [[a]] )

def jacobian_h(state, u, dt, a, b):
    return np.eye(len(state))

class System(object):
    def __init__(self, a, b, initial_state, state_dimension, input_dimension, attack=0, noise=0, sampling=0.1, atol=1e-8, rtol=1e-8, max_step=0.001):
        self.x  = np.array( initial_state ).reshape( state_dimension )
        self.a  = a
        self.b  = b

        self.state_dimension = state_dimension
        self.input_dimension = input_dimension
        self.ts = sampling
        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step
        self.t = 0
        self.attack_indicator = attack
        self.F = lambda x,u:system_flow(x, u, a, b)
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
            measurement = self.x + np.array( [0] ).reshape( self.state_dimension )
        else:
            measurement = self.x + np.array( [0] ).reshape( self.state_dimension )
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
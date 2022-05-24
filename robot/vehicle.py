from cProfile import label
import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
import matplotlib.pyplot as plt 
from numpy.linalg import inv
plt.rcParams.update({'font.size': 15})

class system(object):
    def __init__(self, lf, lr, initial_state, state_dimension, input_dimension, sampling=0.1, atol=1e-8, rtol=1e-8, max_step=0.001):
        self.lf = lf
        self.lr = lr
        self.x  = np.array( initial_state ).reshape( state_dimension )
        self.state_dimension = state_dimension
        self.input_dimension = input_dimension
        self.ts = sampling
        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step
        self.t = 0
    
    def step(self, u):
        dyna = partial(self._flow, u = u)
        sol = solve_ivp(dyna, [self.t, self.t + self.ts], self.x[:,-1].flatten(), max_step=self.max_step, atol=self.atol, rtol=self.rtol, method='RK23', dense_output=True)

        t = sol.t
        x = sol.y
        self.t = t[-1]
        self.x = np.array(x[:, -1]).reshape(self.state_dimension)
        return t, x
    
    def observe(self):
        return self.t, self.x
    
    def get_state(self):
        return self.x

    def _flow(self, t, state, u):
        beta = self.lr/(self.lf + self.lr) * np.tan( u[1] )
        flow = np.zeros( len(state) )
        flow[0] = state[3] * np.cos( state[2] + beta )
        flow[1] = state[3] * np.sin( state[2] + beta )
        flow[2] = state[3]/self.lr * np.sin( beta )
        flow[3] = u[0]
        return flow

    
    def reset(self, initial_state):
        self.x = np.array( initial_state ).reshape(self.state_dimension)
        self.t = 0
        return self.x

class controller(object):
    def __init__(self, state_dimension, input_dimension):
        self.state_dimension = state_dimension
        self.input_dimension = input_dimension
        pass

    def update_u(self, x):
        # output = np.zeros((2,))
        # output[1] = x[2] + np.arctan()
        return np.array([0.1, 0]).reshape( self.input_dimension )

lr = 0.17
lf = 0.15
initial_state = np.array([0, 0, 0, 0])
state_dimension = (len(initial_state), 1)
input_dimension = (2, 1)
ts = 0.1
robot = system(lf, lr, initial_state, state_dimension, input_dimension, ts)
control = controller(state_dimension, input_dimension)

state_store = np.array(initial_state).reshape(state_dimension)
t_system_store     = np.array([0])

u_store     = control.update_u( robot.get_state() )
t_control_store = np.array([0])
for i in range( int(10/ts) ):
    u = control.update_u( robot.get_state() )
    t, x = robot.step( u )
    t_control, _ = robot.observe()

    # data store
    t_system_store = np.hstack( (t_system_store, t) )
    state_store = np.hstack( (state_store, x) )
    u_store = np.hstack( (u_store, u) )
    t_control_store = np.hstack( (t_control_store, t_control) )

fig, ax = plt.subplots( figsize=(8, 3) )
ax.plot( t_system_store, state_store[0:2, :].transpose(), label=["x", "y"] )
ax.set_xlabel( "Time [s]" )
ax.set_ylabel( "Position [m]" )
plt.legend()
plt.savefig(f'Point_3/first_sim_u0_{u[0][0]}_u1_{u[1][0]}.pdf', bbox_inches='tight')

fig, ax = plt.subplots( figsize=(8, 3) )
ax.plot( t_control_store, u_store.transpose(), drawstyle="steps" )

plt.show()

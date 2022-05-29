import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
import matplotlib.pyplot as plt 
from numpy.linalg import inv
from scipy.signal import cont2discrete
from system.system import *
from controller.control import *
from attack_detection.attack_detection import *
from reconfiguration.reconfiguration import *
from reconfiguration.input_estimator import *
plt.rcParams.update({'font.size': 15})

def compute_KPI(y, yd, Ts, state_dimension):
    KPI = np.zeros(state_dimension[0],)
    for i in range( len(y[0]) ):
        for j in range(state_dimension[0]):
            KPI[j] = KPI[j] + Ts * np.abs( y[j, i]-yd[j, i] ) 
    
    return KPI
        
def main():
    # Initial state
    initial_state = np.array([5])
    threshold     = np.array([0.05])


    # state and input dimension
    state_dimension = (len(initial_state), 1)
    input_dimension = (1, 1)
    target_dimension = (len(initial_state), 1)
    ts = 0.1
    target         = np.array([1]).reshape(target_dimension)
    a = -1; b = 1

    # Initialize system 
    estimator_name = "ekf"
    robot    = System( a, b, initial_state, state_dimension, input_dimension, sampling=ts, attack=0, noise=0)
    control  = Controller(target, state_dimension, input_dimension, ts)
    detector = AnomalyDetector(a, b, estimator_name, initial_state, state_dimension, input_dimension, threshold, ts)
    reconf   = Reconfiguration(a, b, ts, input_dimension)
    # input_estimator = InputEsitmator()

    # Initialize Detector
    A_c = np.array( [[a]] )
    B_c = np.array( [[b]] )
    C_c = np.array( [[1]] )
    D_c = np.array( [[0]] )
    d_system = cont2discrete( (A_c, B_c, C_c, D_c), ts )
    a_d = d_system[0][0][0]
    b_d = d_system[1][0][0]
    Q = np.eye(state_dimension[0])
    R = np.eye(state_dimension[0])*10
    P = np.eye(state_dimension[0])
    j_f  = lambda x,u:discrete_jacobian_f(x, u, ts, a_d, b_d)
    j_h  = lambda x,u:discrete_jacobian_h(x, u, ts, a_d, b_d)
    discrete_dynamics = lambda x,u:discrete_system(x, u, a_d, b_d)
    detector.set_estimator( Q, R, P, j_f, j_h, discrete_dynamics )

    # Variables to store  state and time of the system
    state_store        = np.array(initial_state).reshape(state_dimension)
    t_system_store     = np.array([0])

    # Variables to store the residues
    residues_store  = np.array([0]).reshape(state_dimension)
    alarm_store     = np.array([0]).reshape(state_dimension)

    # Variables to store control action and time of cyber world
    u_store     = np.zeros(input_dimension)
    t_control_store = np.array([0])
    
    # Init useful variables
    alarm = detector.verify_alarm()
    x_prediction = initial_state.reshape(state_dimension)

    u_reconfigure = 0
    y_previous = initial_state.reshape(state_dimension)
    y_prediction_previous = initial_state.reshape(state_dimension)
    # Main control loop
    for i in range( int(40/ts) ):
        # Measurement
        t_control, measurement = robot.observe_with_attack()                            # Measure robot states
        
        if i > 0:
            x_estimator, x_prediction = detector.estimate( measurement, uc )            # Predict

        # Anomaly detection
        detector.update_measurement( measurement )                                      # Update detector
        residues = detector.compute_residues()                                          # Compute residues
        alarm    = detector.verify_alarm()                                              # Trigger alarm
        control.update_alarm(alarm)                                                     # Anomaly detection tells the controller if alarm
        detector.estimator.update_alarm(alarm)
        # System

        # Estimate attack
        if sum(alarm) == 0:
            u_reconfigure = 0
        if sum(alarm) > 0 and u_reconfigure == 0:
            u_reconfigure = reconf.reconfigure(measurement, x_prediction, y_previous, y_prediction_previous, t_control)
        # Control computation
        uc = control.update_u( measurement, u_reconfigure )                             # compute controller
        ua = uc + 0
        if t_control > 10 and t_control < 20:
            ua = uc + 2
        t, x = robot.step( ua )                                                         # system step. Store actual system state
        y_previous = measurement + 0
        y_prediction_previous = x_prediction + 0

        # data store
        t_system_store  = np.hstack( (t_system_store, t) )
        state_store     = np.hstack( (state_store, x) )
        u_store         = np.hstack( (u_store, ua) )
        t_control_store = np.hstack( (t_control_store, t_control) )
        residues_store  = np.hstack( (residues_store, residues) )
        alarm_store     = np.hstack( (alarm_store, alarm) )

    # Plotting
    fig, ax = plt.subplots( figsize=(8, 3) )
    ax.plot( t_system_store, state_store.transpose(), label=["$z_1$", "$z_2$", "$z_3$"] )
    ax.set_xlabel( "Time [s]" )
    ax.set_ylabel( "Position [m]" )
    plt.legend()
    # plt.savefig(f'{estimator_name}_attack_detection_state_x_{initial_state[0]}_y_{initial_state[1]}.pdf', bbox_inches='tight')

    fig, ax = plt.subplots( figsize=(8, 3) )
    ax.plot( t_control_store, alarm_store.transpose(), drawstyle="steps", label=["$z_1$", "$z_2$", "$z_3$"] )
    ax.set_xlabel( "Time [s]" )
    ax.set_ylabel( "Alarm" )
    plt.legend()
    # plt.savefig(f'{estimator_name}_attack_detection_alarm_x_{initial_state[0]}_y_{initial_state[1]}.pdf', bbox_inches='tight')

    fig, ax = plt.subplots( figsize=(8, 3) )
    ax.plot( t_control_store, residues_store.transpose(), drawstyle="steps" )
    ax.set_xlabel( "Time [s]" )
    ax.set_ylabel( "Residues $z_3$ [rad]" )
    # plt.savefig(f'{estimator_name}_attack_detection_residues_x_{initial_state[0]}_y_{initial_state[1]}.pdf', bbox_inches='tight')

    fig, ax = plt.subplots( figsize=(8, 3) )
    ax.plot( t_control_store, u_store.transpose(), label=["x", "y"] )
    ax.set_xlabel( "Time [s]" )
    ax.set_ylabel( "Position [m]" )
    plt.legend()
    # plt.savefig(f'{estimator_name}_attack_detection_residues_x_{initial_state[0]}_y_{initial_state[1]}.pdf', bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    main()
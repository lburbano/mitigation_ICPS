from cProfile import label
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
            KPI[j] = KPI[j] + Ts * np.abs( y[j, i]-yd[j] ) 
    
    return KPI
        
def main(attack, reconfiguration):
    # Initial state
    initial_state = np.array([5])
    threshold     = np.array([0.05])


    # state and input dimension
    state_dimension = (len(initial_state), 1)
    input_dimension = (1, 1)
    target_dimension = (len(initial_state), 1)
    ts = 0.01
    target         = np.array([1]).reshape(target_dimension)
    a = -1; b = 1

    # Initialize system 
    estimator_name = "ekf"
    robot    = System( a, b, initial_state, state_dimension, input_dimension, sampling=ts, attack=0, noise=0)
    control  = Controller(target, state_dimension, input_dimension, ts)
    detector = AnomalyDetector(a, b, estimator_name, initial_state, state_dimension, input_dimension, threshold, ts)
    reconf   = Reconfiguration(a, b, ts, input_dimension)
    

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
    h    = lambda x:output_function(x)
    discrete_dynamics = lambda x,u:discrete_system(x, u, a_d, b_d)
    detector.set_estimator( Q, R, P, j_f, j_h, h, discrete_dynamics )

    # Initialize input estimator
    estimator_initial_state = np.array( [initial_state[0], 0] )
    input_estimator = InputEsitmator(a_d, b_d, estimator_initial_state, ts, state_dimension, input_dimension)

    # Variables to store  state and time of the system
    state_store        = np.array(initial_state).reshape(state_dimension)
    t_system_store     = np.array([0])

    # Variables to store the residues
    residues_store  = np.array([0]).reshape(state_dimension)
    alarm_store     = np.array([0]).reshape(state_dimension)

    # Variables to store control action and time of cyber world
    uc_store     = np.zeros(input_dimension)
    ua_store     = np.zeros(input_dimension)
    u_store      = np.zeros(input_dimension)
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
            u_estimation = input_estimator.estimate_u(measurement, uc)

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
        if sum(alarm) > 0:
            if reconfiguration == 2:
                u_reconfigure = u_estimation
            elif reconfiguration == 1 and u_reconfigure == 0:
                u_reconfigure = reconf.reconfigure(measurement, x_prediction, y_previous, y_prediction_previous, t_control)
            
        # Control computation
        uc = control.update_u( measurement, u_reconfigure )                             # compute controller
        ua = uc + 0
        if t_control > 10 and t_control < 30 and attack:
            ua = uc + 2
        t, x = robot.step( ua )                                                         # system step. Store actual system state
        y_previous = measurement + 0
        y_prediction_previous = x_prediction + 0

        # data store
        t_system_store  = np.hstack( (t_system_store, t) )
        state_store     = np.hstack( (state_store, x) )
        uc_store        = np.hstack( (uc_store, uc) )
        ua_store        = np.hstack( (ua_store, ua) )
        t_control_store = np.hstack( (t_control_store, t_control) )
        residues_store  = np.hstack( (residues_store, residues) )
        alarm_store     = np.hstack( (alarm_store, alarm) )
    '''
    KPI = compute_KPI(state_store, target, ts, state_dimension)
    # Plotting
    fig, ax = plt.subplots( figsize=(8, 3) )
    ax.plot( t_system_store, state_store.transpose() )
    ax.set_xlabel( "Time [s]" )
    ax.set_ylabel( "State" )
    # plt.legend()
    plt.grid()
    plt.savefig(f'linear_{estimator_name}_attack_detection_state_x_{initial_state[0]}_attack_{attack}.pdf', bbox_inches='tight')

    fig, ax = plt.subplots( figsize=(8, 3) )
    ax.plot( t_control_store, alarm_store.transpose(), drawstyle="steps" )
    ax.set_xlabel( "Time [s]" )
    ax.set_ylabel( "Alarm" )
    # plt.legend()
    plt.grid()
    plt.savefig(f'linear_{estimator_name}_attack_detection_alarm_x_{initial_state[0]}_attack_{attack}.pdf', bbox_inches='tight')

    fig, ax = plt.subplots( figsize=(8, 3) )
    ax.plot( t_control_store, residues_store.transpose(), drawstyle="steps" )
    ax.set_xlabel( "Time [s]" )
    ax.set_ylabel( "Residues" )
    plt.grid()
    plt.savefig(f'linear_{estimator_name}_attack_detection_residues_x_{initial_state[0]}_attack_{attack}.pdf', bbox_inches='tight')

    fig, ax = plt.subplots( figsize=(8, 3) )
    ax.plot( t_control_store, ua_store.transpose() )
    # ax.plot( t_control_store, ua_store.transpose() )
    ax.set_xlabel( "Time [s]" )
    ax.set_ylabel( "Control action" )
    # plt.legend()
    plt.grid()
    plt.savefig(f'linear_{estimator_name}_attack_detection_uc_x_{initial_state[0]}_attack_{attack}.pdf', bbox_inches='tight')
    '''
    return t_system_store, state_store.transpose()
    # plt.show()

def main_multiple_sim():
    attack = 1
    reconfiguration = 0
    t_attack, states_attack = main(attack, reconfiguration)

    attack = 1
    reconfiguration = 1
    t_reconfiguration, states_reconfiguration = main(attack, reconfiguration)

    attack = 1
    reconfiguration = 2
    t_reconfiguration_2, states_reconfiguration_2 = main(attack, reconfiguration)

    attack = 0
    reconfiguration = 0
    t_no_attack, states_no_attack = main(attack, reconfiguration)

    fig, ax = plt.subplots( figsize=(8, 3) )
    ax.plot( t_no_attack, states_no_attack, label="NA" )
    ax.plot( t_attack, states_attack, label="A - NR" )
    ax.plot( t_reconfiguration, states_reconfiguration, label="A - R1" )
    ax.plot( t_reconfiguration_2, states_reconfiguration_2, label="A - R2" )
    
    plt.legend()
    ax.set_xlabel( "Time [s]" )
    ax.set_ylabel( "State" )
    plt.grid()
    plt.savefig(f'linear_system_reconfiguration.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main_multiple_sim()
from charset_normalizer import detect
import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
import matplotlib.pyplot as plt 
from numpy.linalg import inv
import random
from system_robot.system import *
from controller.control import *
from attack_detection.attack_detection import *
from reconfiguration.reconfiguration import *
from reconfiguration.input_estimator import *
plt.rcParams.update({'font.size': 15})


        
def main(attack, reconfiguration):
    # Initial state
    initial_state = np.array([0, 0, 0])
    threshold     = np.array([0.01, 0.01, 0.01])


    # state and input dimension
    state_dimension = (len(initial_state), 1)
    input_dimension = (2, 1)
    target_dimension = (len(initial_state)-1, 1)
    ts = 0.01
    target         = np.array([0.5, 0.1]).reshape(target_dimension)
    

    # Initialize system 
    estimator_name = "ekf"
    robot    = System( initial_state, state_dimension, input_dimension, sampling=ts, attack=0, noise=0)
    control  = Controller(target, state_dimension, input_dimension, ts)
    detector = AnomalyDetector(estimator_name, initial_state, state_dimension, input_dimension, threshold, ts)

    jac_A = lambda x, u:jacobian_A(x, u, ts)
    jac_B = lambda x, u:jacobian_B(x, u, ts)
    reconf = Reconfiguration(jac_A, jac_B, ts, input_dimension)
    
    # Initialize Detector
    Q = np.eye(state_dimension[0])
    R = np.eye(state_dimension[0])*10
    P = np.eye(state_dimension[0])
    j_f  = lambda x,u:discrete_jacobian_f(x, u, ts)
    j_h  = lambda x,u:discrete_jacobian_h(x, u, ts)
    h    = lambda x:output_function(x)
    discrete_dynamics = lambda x,u:discrete_system(x, u, ts)
    detector.set_estimator( Q, R, P, j_f, j_h, h, discrete_dynamics )

    # Initialize input estimator
    initial_state_augmented = estimator_initial_state = np.array( [initial_state[0], initial_state[1], initial_state[2], 0, 0] )
    j_f_augmented = lambda x, u:augmented_discrete_jacobian_f(x, u, ts)
    j_h_augmented = lambda x, u:augmented_discrete_jacobian_h(x, u, ts)
    dynamics_augmented = lambda x, u:augmented_discrete_system(x, u, ts)
    output_augmented = lambda x:augmented_output_function(x)
    input_estimator = InputEsitmator(j_f_augmented, j_h_augmented, output_augmented, dynamics_augmented, initial_state_augmented, ts, state_dimension, input_dimension)

    # Variables to store  state and time of the system
    state_store        = np.array(initial_state).reshape(state_dimension)
    t_system_store     = np.array([0])

    # Variables to store the residues
    residues_store  = np.array([0, 0, 0]).reshape(state_dimension)
    alarm_store     = np.array([0, 0, 0]).reshape(state_dimension)

    # Variables to store control action and time of cyber world
    u_store     = np.zeros(input_dimension)
    t_control_store = np.array([0])
    
    # Init useful variables
    alarm = detector.verify_alarm()
    x_prediction = initial_state.reshape(state_dimension)
    u_reconfigure = np.array( np.zeros( input_dimension ) ).reshape(input_dimension)

    y_previous = initial_state.reshape(state_dimension)
    y_prediction_previous = initial_state.reshape(state_dimension)
    # Main control loop
    for i in range( int(100/ts) ):
        # Measurement
        t_control, measurement = robot.observe_with_attack()                # Measure robot states
        
        if i > 0:
            x_estimator, x_prediction = detector.estimate( measurement, uc ) # Predict
            x_estimator_augmented = input_estimator.estimate_u(measurement, uc)
            u_estimated = x_estimator_augmented[3:]
            u_estimated = u_estimated.reshape(input_dimension)

        # Anomaly detection
        detector.update_measurement( measurement )                          # Update detector
        residues = detector.compute_residues()                              # Compute residues
        alarm    = detector.verify_alarm()                                  # Trigger alarm
        control.update_alarm(alarm)                                         # Anomaly detection tells the controller if alarm
        detector.estimator.update_alarm(alarm)
        # Control computation
        if sum(alarm) == 0:
            u_reconfigure = np.array( np.zeros( input_dimension ) ).reshape(input_dimension)
        if sum(alarm) > 0:
            # data = measurement
            if reconfiguration == 1 and sum( np.abs(u_reconfigure) ) == 0:
                u_reconfigure = reconf.reconfigure( uc, measurement, x_prediction, y_previous, y_prediction_previous, t_control )
            elif reconfiguration == 2:
                u_reconfigure = u_estimated
        # Control computation
        uc = control.update_u( measurement, u_reconfigure )                             # compute controller
        ua = uc + 0
        if t_control > 10 and t_control < 60 and attack:
            ua = uc + np.array( [2, 0] ).reshape(input_dimension)
        t, x = robot.step( ua )                                                         # system step. Store actual system state
        y_previous = measurement + 0
        y_prediction_previous = x_prediction + 0

        # data store
        t_system_store  = np.hstack( (t_system_store, t) )
        state_store     = np.hstack( (state_store, x) )
        u_store         = np.hstack( (u_store, uc) )
        t_control_store = np.hstack( (t_control_store, t_control) )
        residues_store  = np.hstack( (residues_store, residues) )
        alarm_store     = np.hstack( (alarm_store, alarm) )
    '''
    # Plotting
    fig, ax = plt.subplots( figsize=(8, 3) )
    ax.plot( t_system_store, state_store[0:3, :].transpose(), label=["$z_1$", "$z_2$", "$z_3$"] )
    ax.set_xlabel( "Time [s]" )
    ax.set_ylabel( "Position [m]" )
    plt.legend()
    # plt.savefig(f'{estimator_name}_attack_detection_state_x_{initial_state[0]}_y_{initial_state[1]}.pdf', bbox_inches='tight')
    
    fig, ax = plt.subplots( figsize=(8, 3) )
    ax.plot( t_control_store, sum(alarm_store).transpose()/(0.0001+sum(alarm_store).transpose()), drawstyle="steps")
    ax.set_xlabel( "Time [s]" )
    ax.set_ylabel( "Alarm" )
    plt.grid()
    # plt.legend()
    plt.savefig(f'{estimator_name}_attack_detection_alarm_x_{initial_state[0]}_y_{initial_state[1]}.pdf', bbox_inches='tight')
    
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
    '''
    return t_system_store, state_store.transpose()

def main_multiple_sim():
    attack = 0
    reconfiguration = 0
    time_no_attack, state_no_attack = main(attack, reconfiguration)

    attack = 1
    reconfiguration = 0
    time_attack_no_reconfiguration, state_attack_no_reconfiguration = main(attack, reconfiguration)

    attack = 1
    reconfiguration = 1
    time_attack_reconfiguration_1, state_attack_reconfiguration_1 = main(attack, reconfiguration)

    attack = 1
    reconfiguration = 2
    time_attack_reconfiguration_2, state_attack_reconfiguration_2 = main(attack, reconfiguration)

    fig, ax = plt.subplots( figsize=(8, 3) )
    ax.plot( time_no_attack, state_no_attack[:, 0], label="NA" )
    ax.plot( time_attack_no_reconfiguration, state_attack_no_reconfiguration[:, 0], label="A - NR" )
    ax.plot( time_attack_reconfiguration_1, state_attack_reconfiguration_1[:, 0], label="A - R1" )
    ax.plot( time_attack_reconfiguration_2, state_attack_reconfiguration_2[:, 0], label="A - R2" )
    plt.legend(ncol=2)
    ax.set_xlabel( "Time [s]" )
    ax.set_ylabel( "Position $z_1$" )
    plt.grid()
    # plt.savefig(f'attack_mitigation_w_robot_z1_both.pdf', bbox_inches='tight')

    fig, ax = plt.subplots( figsize=(8, 3) )
    ax.plot( time_no_attack, state_no_attack[:, 1], label="NA" )
    ax.plot( time_attack_no_reconfiguration, state_attack_no_reconfiguration[:, 1], label="A - NR" )
    ax.plot( time_attack_reconfiguration_1, state_attack_reconfiguration_1[:, 1], label="A - R" )
    ax.plot( time_attack_reconfiguration_2, state_attack_reconfiguration_2[:, 1], label="A - R2" )
    plt.legend(ncol=2)
    ax.set_xlabel( "Time [s]" )
    ax.set_ylabel( "Position $z_2$" )
    # ax.set_ylim([-2, 4])
    plt.grid()
    # plt.savefig(f'attack_mitigation_w_robot_z2_both.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main_multiple_sim()
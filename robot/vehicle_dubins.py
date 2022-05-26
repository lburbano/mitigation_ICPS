import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
import matplotlib.pyplot as plt 
from numpy.linalg import inv
import random
from system.system import *
from controller.control import *
from attack_detection.attack_detection import *
from reconfiguration.reconfiguration import *
plt.rcParams.update({'font.size': 15})


        
def main():
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
    reconf = Reconfiguration(ts, input_dimension)

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
    for i in range( int(40/ts) ):
        # Measurement
        t_control, measurement = robot.observe_with_attack()                # Measure robot states
        
        if i > 0:
            x_estimator, x_prediction = detector.estimate( measurement, uc ) # Predict

        # Anomaly detection
        detector.update_measurement( measurement )                          # Update detector
        residues = detector.compute_residues()                              # Compute residues
        alarm    = detector.verify_alarm()                                  # Trigger alarm
        control.update_alarm(alarm)                                         # Anomaly detection tells the controller if alarm
        detector.estimator.update_alarm(alarm)
        # Control computation
        if sum(alarm) == 0:
            u_reconfigure = np.array( np.zeros( input_dimension ) ).reshape(input_dimension)
        if sum(alarm) > 0 and not any(np.abs(u_reconfigure) > 0):
            data = measurement
            u_reconfigure = reconf.reconfigure(uc, measurement, x_prediction, y_previous, y_prediction_previous, ts)
        # Control computation
        uc = control.update_u( measurement, u_reconfigure )                             # compute controller
        ua = uc + 0
        if t_control > 10 and t_control < 20:
            ua = uc + np.array( [1, 1] ).reshape(input_dimension)
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

    # Plotting
    fig, ax = plt.subplots( figsize=(8, 3) )
    ax.plot( t_system_store, state_store[0:3, :].transpose(), label=["$z_1$", "$z_2$", "$z_3$"] )
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
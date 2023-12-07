import casadi
import numpy as np

def example_9_10():
    """Example 9.10: Open-loop planning in a deterministic environment.
    
    We attempt to find a path around a circular obstacle. This implementation
    uses the CasADi interface to the Ipopt solver. 

    Joel A E Andersson, undefined., et al. "CasADi – A software framework for
    nonlinear optimization and optimal control," in Mathematical Programming
    Computation, vol. 11, no. 1, pp. 1–36, 2019.
    
    A. Wächter and L.T. Biegler, "On the Implementation of an Interior-Point
    Filter Line-Search Algorithm for Large-Scale Nonlinear Programming,"
    Mathematical Programming, vol. 106, no.1, pp. 25-27, 2005.
    """
    d = 10                                   # Horizon
    current_state = np.zeros(4)              # Start at the origin w/ 0 velocity
    goal = np.array([10.0, 10.0, 0.0, 0.0])  # End at (10, 10) w/ 0 velocity
    obstacle = np.array([3.0, 4.0])          # Circle at (3, 4) w/ radius 2

    opti = casadi.Opti()

    # Variables
    s = opti.variable(4, d)  # State:  [x_pos, y_pos, x_vel, y_vel]
    a = opti.variable(2, d)  # Action: [x_accel, y_accel]

    # Acceleration bounds
    opti.subject_to(opti.bounded(-1, a, 1))

    # Velocity update
    opti.subject_to([s[2 + j, i] == s[2 + j, i - 1] + a[j, i - 1] for i in range(1, d) for j in range(2)])
    # Position update
    opti.subject_to([s[j, i] == s[j, i - 1] + s[2 + j, i - 1] for i in range(1, d) for j in range(2)])
    
    # Alternative Way to write the velocity & position updates: State-space matrices s = As + Ba
    # A = np.array([[1, 0, 1, 0],
    #               [0, 1, 0, 1],
    #               [0, 0, 1, 0],
    #               [0, 0, 0, 1]])
    # B = np.array([[0, 0],
    #               [0, 0],
    #               [1, 0],
    #               [0, 1]])
    # opti.subject_to([s[:, i] == A @ s[:, i - 1] + B @ a[:, i - 1] for i in range(1, d)])
    
    # Initial condition
    opti.subject_to([s[:, 0] == current_state])
    # Obstacle
    opti.subject_to([casadi.sumsqr(s[:2, i] - obstacle) >= 4 for i in range(1, d)])

    # Objective function
    opti.minimize(100 * casadi.sumsqr(s[:, d - 1] - goal) + casadi.sumsqr(a))
    
    # Utilize the Ipopt solver
    options = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver("ipopt", options) 

    # Optimize
    sol = opti.solve()

    print("First action: ", sol.value(a[:, 0]))

example_9_10()

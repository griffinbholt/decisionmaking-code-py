import numpy as np
import sys; sys.path.append('../')

from ch7 import LinearQuadraticProblem

# Example 7.4: Solving a finite horizon MDP with linear transition function & quadratic award
def example_7_4():
    t_step = 1
    Ts = np.array([[1, t_step], [0, 1]])
    Ta = np.array([[0.5*(t_step**2)], [t_step]])
    Rs = -np.eye(2)
    Ra = -np.array([[0.5]])
    h_max = 5

    lqp = LinearQuadraticProblem(Ts, Ta, Rs, Ra, h_max)
    opt_policies = lqp.solve()
    
    s = np.array([-10, 0])
    for i in range(h_max):
        a = opt_policies[i](s)
        s = (Ts @ s) + (Ta @ a)
        print(s)

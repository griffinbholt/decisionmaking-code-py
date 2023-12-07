import sys; sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from ch08 import ApproximateValueIteration, LinearRegressionValueFunction
from problems.MountainCarMDP import MountainCar

def example_8_1(): # TODO - Need to fix how V is generated
    """Example 8.1: Simplex interpolation in three dimensions"""
    s = np.array([0.3, 0.7, 0.2])  # State

    # Vertices of the Simplex
    V = np.zeros((len(s), 1))
    for i, s_prime in enumerate(reversed(np.argsort(s))):
        vertex = V[:, i] + np.eye(len(s))[:, s_prime]
        V = np.hstack([V, vertex[:, np.newaxis]])
    
    w = np.zeros(V.shape[1])
    w[1:] = np.linalg.solve(V[:, 1:], s) # Determine values of last three weights
    w[0] = 1 - np.sum(w[1:])             # Obtain w1 by enforcing weights sum to 1

    print("Weights: w = ", w)


def example_8_2():
    pass  # TODO


def example_8_3():
    """
    Example 8.3: Linear regression using Fourier bases used to approximate the
    value function for the mountain car problem (see appendix F.4 in the textbook).
    Value functions (top plots) and resulting policies (bottom plots) are shown.
    The globally approximated value function is a poor fit despite using eighth-order
    Fourier basis functions. The resulting approximate policy is not a close
    approximation to the expert policy. The small time step in the mountain car
    problem causes even small changes in the value function landscape to affect
    the policy. Optimal utility functions often have complex geometries that can
    be difficult to capture with global basis functions.
    """
    P = MountainCar
    ss = np.array([[x, v] for x in np.linspace(P.STATE_MINS[0], P.STATE_MAXS[0], 100) for v in np.linspace(P.STATE_MINS[1], P.STATE_MAXS[1], 100)])
    def beta(s: np.ndarray, deg=8) -> np.ndarray:
        def b_s(x, i, T): return np.sin(2*np.pi*i*x / T)
        def b_c(x, i, T): return np.cos(2*np.pi*i*x / T)

        x, v = s
        T_x = MountainCar.STATE_MAXS[0] - MountainCar.STATE_MINS[0]
        T_v = MountainCar.STATE_MAXS[1] - MountainCar.STATE_MINS[1]

        return np.concatenate([[1], 
                               np.array([b_s(x, i + 1, T_x) for i in range(deg)]), 
                               np.array([b_c(x, i + 1, T_x) for i in range(deg)]),
                               np.array([b_s(v, i + 1, T_v) for i in range(deg)]), 
                               np.array([b_c(v, i + 1, T_v) for i in range(deg)])])

    U_theta = LinearRegressionValueFunction(beta=beta, theta=np.zeros(33))
    ApproximateValueIteration(U_theta, ss, k_max=1000).solve(P)

    # TODO - Talk to Mykel - because I can't iterate over states, I don't know how to do a backup/lookahead in ApproximateValueIteration

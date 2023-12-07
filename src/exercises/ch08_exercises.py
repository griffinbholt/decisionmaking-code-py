import sys; sys.path.append('../')

import numpy as np

from ch08 import NearestNeighborValueFunction

def exercise_8_4():
    S = np.array([[4, 5], [2, 6], [-1, -1]])
    U = np.array([2, 10, 30])
    s = np.array([1, 2])

    for ord in [1, 2, np.inf]:
        def d(s, s_prime, ord=ord): return np.linalg.norm(s - s_prime, ord=ord)
        nn_value_func = NearestNeighborValueFunction(2, d, S, U)
        value = nn_value_func(s)
        print("L" + str(ord) + " norm:\t", str(value))


def exercise_8_7():
    """Example 8.7: Bilinear interpolation in two dimensions"""
    states = np.array([[0, 5], [0, 25], [1, 5], [1, 25]])  # known states
    s = np.array([0.7, 10])                                # new state
    
    all_states = np.vstack([states, s])
    x = np.unique(all_states[:, 0])  # (x_1, x_new, x_2)
    y = np.unique(all_states[:, 1])  # (y_1, y_new, y_2)
 
    numers = np.flip(np.outer(np.diff(x), np.diff(y)).flatten())
    denom = np.ptp(x) * np.ptp(y)
    weights = numers / denom
    
    print("Weights: w = ", weights)


def exercise_8_8():
    """Example 8.8: Simplex interpolation in three dimensions"""
    s = np.array([0.4, 0.95, 0.6])  # State

    # Vertices of the Simplex
    V = np.zeros((len(s), 1))
    for i, s_prime in enumerate(reversed(np.argsort(s))):
        vertex = V[:, i] + np.eye(len(s))[:, s_prime]
        V = np.hstack([V, vertex[:, np.newaxis]])
    
    w = np.zeros(V.shape[1])
    w[1:] = np.linalg.solve(V[:, 1:], s) # Determine values of last three weights
    w[0] = 1 - np.sum(w[1:])             # Obtain w1 by enforcing weights sum to 1

    print("Weights: w = ", w)

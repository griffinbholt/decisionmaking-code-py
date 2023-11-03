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

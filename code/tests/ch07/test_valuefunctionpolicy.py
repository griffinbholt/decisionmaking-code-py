import sys; sys.path.append('./code/'); sys.path.append('../../')

import numpy as np

from ch07 import MDP, ValueFunctionPolicy
from ThreeTileStraightlineHexworld import gamma, S, A, T, R, TR, policy

class TestValueFunctionPolicy():
    P = MDP(gamma, S, A, T, R, TR)
    U = np.array([1.425, 2.128, 10, 0])

    def test(self):  # See Exercise 7.5
        policy = ValueFunctionPolicy(self.P, self.U)
        for state in self.P.S:
            assert(policy(state) == "east")
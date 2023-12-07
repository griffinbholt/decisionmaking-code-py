import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np

from ch07 import ValueFunctionPolicy
from problems.HexWorldMDP import ThreeTileStraightLineHexWorld, action_to_idx

class TestValueFunctionPolicy():
    P = ThreeTileStraightLineHexWorld
    U = np.array([1.425, 2.128, 10, 0])

    def test(self):  # See Exercise 7.5
        policy = ValueFunctionPolicy(self.P, self.U)
        for state in self.P.S:
            assert(policy(state) == action_to_idx["east"])
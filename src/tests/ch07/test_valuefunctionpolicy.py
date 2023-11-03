import numpy as np
import problems.ThreeTileStraightLineHexworld
import sys; sys.path.append('./code/'); sys.path.append('../../')

from ch07 import MDP, ValueFunctionPolicy

class TestValueFunctionPolicy():
    P = problems.ThreeTileStraightLineHexworld.P
    U = np.array([1.425, 2.128, 10, 0])

    def test(self):  # See Exercise 7.5
        policy = ValueFunctionPolicy(self.P, self.U)
        for state in self.P.S:
            assert(policy(state) == "east")
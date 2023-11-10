import sys; sys.path.append('./src/'); sys.path.append('../'); sys.path.append('../../')

import numpy as np

from problems.HexWorldMDP import StraightLineHexWorld
from ch14 import adversarial


class TestAdversarial():
    def test(self):
        P = StraightLineHexWorld
        def policy(s): return 0
        assert(adversarial(P, policy, 0.5).T(0, 0, 0) == 1.0)
import sys; sys.path.append('./src/'); sys.path.append('../'); sys.path.append('../../')

import numpy as np

from problems import ThreeTileStraightLineHexworld
from ch14 import adversarial


class TestAdversarial():
    def test(self):
        P = adversarial(ThreeTileStraightLineHexworld.P,
                        ThreeTileStraightLineHexworld.init_policy,
                        lam=0.1)
        print(P.gamma)
        print(P.A)
        print(P.S)
        print([[P.R(s, a) for a in P.A] for s in P.S])

test = TestAdversarial()
test.test()
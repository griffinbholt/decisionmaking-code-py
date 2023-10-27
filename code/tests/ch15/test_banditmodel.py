import sys; sys.path.append('./code/'); sys.path.append('../../')

import numpy as np

from scipy.stats import beta

from ch15 import BanditModel


class TestBanditModel():
    def test(self, tol=1e-10):
        model = BanditModel(B=[beta(1, 1), beta(1, 1)])
        assert(model.B[0].args[0] == 1)
        assert(model.B[0].args[1] == 1)
        assert(model.B[1].args[0] == 1)
        assert(model.B[1].args[1] == 1)
        
        model.update(a=0, r=1) # First Arm, Win
        model.update(a=1, r=1) # Second Arm, Win
        model.update(a=1, r=1) # Second Arm, Win
        model.update(a=1, r=1) # Second Arm, Win
        model.update(a=1, r=1) # Second Arm, Win
        model.update(a=1, r=0) # Second Arm, Loss
        
        assert(model.B[0].args[0] == 2)
        assert(model.B[0].args[1] == 1)
        assert(model.B[1].args[0] == 5)
        assert(model.B[1].args[1] == 2)

        assert(np.abs(model.B[0].mean() - (2/3)) < tol)
        assert(np.abs(model.B[1].mean() - (5/7)) < tol)

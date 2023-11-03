import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np

from ch17 import IncrementalEstimate

class TestIncrementalEstimate():
    m = 5    # Size of random vector
    n = 500  # Number of samples
    x = np.random.random(size=(n, m))
    tol = 1e-14

    def test_update_scalar(self):
        incr_est = IncrementalEstimate(mu=self.x[0, 0],
                                       alpha=lambda m: 1/m,
                                       m=1)
        for i in range(1, self.n):
            incr_est.update(self.x[i, 0])
            assert(np.abs(incr_est.mu - np.mean(self.x[:(i + 1), 0])) < self.tol)

    def test_update_vector(self):
        incr_est = IncrementalEstimate(mu=self.x[0].copy(),
                                       alpha=lambda m: 1/m,
                                       m=1)
        for i in range(1, self.n):
            incr_est.update(self.x[i])
            assert(np.all(np.abs(incr_est.mu - np.mean(self.x[:(i + 1)], axis=0)) < self.tol))

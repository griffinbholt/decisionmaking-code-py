import numpy as np
import sys; sys.path.append('./code/'); sys.path.append('../../')
from scipy.stats import multivariate_normal

from ch03 import MultivariateGaussianInference


class TestMultivariateGaussianInference():
    def test(self, tol=1e-15):
        # Joint Distribution
        mean = np.array([0, 1])
        cov = np.array([[3, 1], [1, 2]])
        D = multivariate_normal(mean, cov)

        # Query: x_1
        query = np.array([0])

        # Evidence: x_2 = 2.0
        evidence_vars = np.array([1])
        evidence = np.array([2.0])

        M = MultivariateGaussianInference()
        cond_distrib = M.infer(D, query, evidence_vars, evidence)

        assert cond_distrib.mean.shape == (1,)
        assert cond_distrib.cov.shape == (1, 1)
        assert np.abs(cond_distrib.mean[0] - 0.5) < tol
        assert np.abs(cond_distrib.cov[0, 0] - 2.5) < tol

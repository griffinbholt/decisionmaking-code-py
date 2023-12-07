import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np

from ch08 import ApproximateValueIteration, NearestNeighborValueFunction, LocallyWeightedValueFunction, MultilinearValueFunction, SimplexValueFunction, LinearRegressionValueFunction
from problems.HexWorldMDP import StraightLineHexWorld

class TestApproximateValueFunctions():
    tol = 1e-5

    def test_nearest_neighbor(self):  # See Exercise 8.4
        S = np.array([[4, 5], [2, 6], [-1, -1]])
        U = np.array([2, 10, 30])
        s = np.array([1, 2])

        norms = [1, 2, np.inf]
        exp_answers = [20, 20, 16]
        for i in range(len(norms)):
            def d(s, s_prime, ord=norms[i]): return np.linalg.norm(s - s_prime, ord=ord)
            nn_value_func = NearestNeighborValueFunction(2, d, S, U)
            value = nn_value_func(s)
            assert(value == exp_answers[i])

    def test_locally_weighted(self):
        def k(s, s_prime): return np.linalg.norm(s - s_prime, ord=2)**2
        S = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        U_theta = LocallyWeightedValueFunction(k, S, theta=[0.0, 0.1, 0.3, 0.2])
        assert(np.abs(U_theta(np.array([0.0, 0.0])) - 0.225) < self.tol)
        assert(np.abs(U_theta(np.array([3.0, 1.0])) - 0.125) < self.tol)

    def test_multilinear_interpolation(self):
        self.run_interpolation_test(MultilinearValueFunction)
    
    def test_simplex_interpolation(self):
        self.run_interpolation_test(SimplexValueFunction)
    
    def run_interpolation_test(self, Interpolation):
        so = np.array([-3.0, -3.0])
        us = np.array([[0.0, 1.0], [2.0, 3.0]])
        delta = np.array([6.0, 6.0])
        
        def interpolation(us, so, delta, s): return Interpolation(so, delta, us)(s)

        v = interpolation(us, so, delta, [-3.0,-3.0])
        assert(np.isclose(v, 0.0, atol=self.tol))
        v = interpolation(us, so, delta, [3.0,3.0])
        assert(np.isclose(v, 3.0, atol=self.tol))
        v = interpolation(us, so, delta, [0.0,-3.0])
        assert(np.isclose(v, 1.0, atol=self.tol))
        v = interpolation(us, so, delta, [0.0,3.0])
        assert(np.isclose(v, 2.0, atol=self.tol))
        v = interpolation(us, so, delta, [0.0,0.0])
        assert(np.isclose(v, (0 + 1 + 2 + 3)/4.0, atol=self.tol))

    def test_linear_regression(self):
        ss = [0, 2, 4, 6, 7]  # NOTE: 7 is terminal state
        def beta(s): return np.array([1.0, s, s**2, s == 7])

        U_theta = LinearRegressionValueFunction(beta, np.zeros(4))
        policy = ApproximateValueIteration(U_theta, ss, k_max=50).solve(StraightLineHexWorld)

        # Policy should always be east
        for s in StraightLineHexWorld.S:
            assert(policy(s) == 0)
        
        # Regression test
        theta_expected = np.array([2.3899132417230033, 0.7603331881146517, 0.08448136415721541, -11.851832402229398])
        assert(np.all(np.isclose(U_theta.theta - theta_expected, 0.0, atol=self.tol)))

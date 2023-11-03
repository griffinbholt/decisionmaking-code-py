import sys; sys.path.append('./code/'); sys.path.append('../../')

import numpy as np

from ch08 import NearestNeighborValueFunction, LocallyWeightedValueFunction, MultilinearValueFunction, SimplexValueFunction, LinearRegressionValueFunction


class TestApproximateValueFunctions():
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
        pass  # TODO

    def test_multilinear_interpolation(self):
        pass  # TODO

    def test_simplex_interpolation(self):
        pass  # TODO

    def test_linear_regression(self):
        pass  # TODO

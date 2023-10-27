import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Callable

from ch07 import MDP, OfflinePlanningMethod, ValueFunctionPolicy
from convenience import normalize


class ApproximateValueFunction(ABC):
    @abstractmethod
    def __call__(self, s: Any) -> float:
        pass

    @abstractmethod
    def fit(self, S: list[Any], U: np.ndarray):
        pass


class ApproximateValueIteration(OfflinePlanningMethod):
    def __init__(self, init_U_theta: ApproximateValueFunction, S: list[Any], k_max: int):
        self.init_U_theta = init_U_theta  # initial parameterized value function that supports fit
        self.S = S                        # set of discrete states for performing backups
        self.k_max = k_max                # maximum number of iterations

    def solve(self, P: MDP) -> Callable[[Any], Any]:
        U_theta = self.init_U_theta.copy()  # TODO - Test depth of this copy
        for _ in range(self.k_max):
            U = np.array([P.backup(U_theta, s) for s in self.S])
            U_theta.fit(self.S, U)
        return ValueFunctionPolicy(P, U_theta)


class NearestNeighborValueFunction(ApproximateValueFunction):
    def __init__(self, k: int, d: Callable[[Any, Any], float], S: list[Any], theta: np.ndarray):
        self.k = k          # number of neighbors
        self.d = d          # distance function d(s, s')
        self.S = S          # set of discrete states
        self.theta = theta  # vector of values at states in S

    def __call__(self, s: Any) -> float:
        distances = np.ndarray([self.d(s, s_prime) for s_prime in self.S])
        indices = np.argsort(distances)[:self.k]
        return np.mean([self.theta[i] for i in indices])

    def fit(self, S: list[Any], U: np.ndarray):
        self.theta = U


class LocallyWeightedValueFunction(ApproximateValueFunction):
    def __init__(self, k: Callable[[Any], Any], S: list[Any], theta: np.ndarray):
        self.k = k          # kernel function k(s, s′)
        self.S = S          # set of discrete states
        self.theta = theta  # vector of values at states in S

    def __call__(self, s: Any) -> float:
        w = normalize(np.array([self.k(s, s_prime) for s_prime in self.S]), ord=1)
        return np.dot(self.theta, w)

    def fit(self, S: list[Any], U: np.ndarray):
        self.theta = U


class InterpolationValueFunction(ApproximateValueFunction):
    """
    A parent class for interpolation methods, used to estimate the value
    of state vector `s` for known state values `theta` over a grid defined by
    lower-left vertex `o` and vector of widths `delta`.

    Vertices of the grid can all be written `o + delta * i` for some nonnegative
    integral vector `i`.
    """
    def __init__(self, o: np.ndarray, delta: np.ndarray, theta: np.ndarray):
        self.o = o          # position of lower-left corner
        self.delta = delta  # vector of widths
        self.theta = theta  # vector of values at states in S

    @abstractmethod
    def __call__(self, s: np.ndarray) -> float:
        pass

    def fit(self, S: list[Any], U: np.ndarray):
        self.theta = U


class MultilinearValueFunction(InterpolationValueFunction):
    """Multilinear Interpolation"""
    def __call__(self, s: np.ndarray) -> float:
        Delta = (s - self.o) / self.delta
        # Multidimensional index of lower-left cell
        i = np.minimum(np.floor(Delta).astype(int) + 1, np.array(self.theta.shape) - 1)
        vertex_index = np.empty_like(i)
        d = len(s)
        u = 0.0
        for vertex in range(2**d):
            weight = 1.0
            for j in range(d):
                # Check whether jth bit is set
                if vertex & (1 << j) > 0:
                    vertex_index[j] = i[j] + 1
                    weight *= Delta[j] - i[j] + 1
                else:
                    vertex_index[j] = i[j]
                    weight *= i[j] - Delta[j]
            u += self.theta[tuple(vertex_index)] * weight
        return u


class SimplexValueFunction(InterpolationValueFunction):
    """Simplex Interpolation"""
    def __call__(self, s: np.ndarray) -> float:
        Delta = (s - self.o) / self.delta
        # Multidimensional index of upper-right cell
        i = np.minimum(np.floor(Delta).astype(int) + 1, np.array(self.theta.shape) - 1) + 1
        u = 0.0
        s_prime = (s - (self.o + (self.delta * (i - 2)))) / self.delta
        p = np.argsort(s_prime)  # increasing order
        w_tot = 0.0
        for j in p:
            w = s_prime[j] - w_tot
            u += w * self.theta[tuple(i)]
            i[j] -= 1
            w_tot += w
        u += (1 - w_tot) * self.theta[tuple(i)]
        return u


class LinearRegressionValueFunction(ApproximateValueFunction):
    def __init__(self, beta: Callable[[Any], np.ndarray], theta: np.ndarray):
        self.beta = beta    # basis vector function
        self.theta = theta  # vector of parameters

    def __call__(self, s: Any) -> float:
        return np.dot(self.theta, self.beta(s))

    def fit(self, S: list[Any], U: np.ndarray):
        X = np.array([self.beta(s) for s in S])
        self.theta = np.linalg.solve(X.T @ X, X.T @ U)

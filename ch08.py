import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Callable

from ch07 import MDP, SolutionMethod, ValueFunctionPolicy

class ApproximateSolutionMethod(SolutionMethod):
    pass

class ApproximateValueFunction(ABC):
    @abstractmethod
    def __call__(self, s: Any) -> float:
        pass

    @abstractmethod
    def fit(self, S: list[Any], U: np.ndarray):
        pass

class ApproximateValueIteration(ApproximateSolutionMethod):
    def __init__(self, U_theta: ApproximateValueFunction, S: list[Any], k_max: int):
        self.init_U_theta = init_U_theta # initial parameterized value function that supports fit
        self.S = S # set of discrete states for performing backups
        self.k_max = k_max # maximum number of iterations

    def solve(P: MDP):
        U_theta = self.init_U_theta.copy() # TODO - Test depth of this copy
        for _ in range(self.k_max):
            U = np.array([P.backup(U_theta, s) for s in self.S])
            U_theta.fit(S, U)
        return ValueFunctionPolicy(P, U_theta)

class NearestNeighborValueFunction(ApproximateValueFunction):
    def __init__(self, k: int, d: Callable[[Any, Any], float], S: list[Any], theta: np.ndarray):
        self.k = k # number of neighbors
        self.d = d # distance function d(s, s')
        self.S = S # set of discrete states
        self.theta = theta # vector of values at states in S

    def __call__(self, s: Any) -> float:
        distances = np.ndarray([self.d(s, s_prime) for s_prime in self.S])
        indices = np.argsort(distances)[:self.k]
        return np.mean([self.theta[i] for i in indices])
    
    def fit(self, S: list[Any], U: np.ndarray):
        self.theta = U

class LocallyWeightedValueFunction(ApproximateValueFunction):
    def __init__(self, k: Callable[[Any], Any], S: list[Any], theta: np.ndarray):
        self.k = k # kernel function k(s, s′)
        self.S = S # set of discrete states
        self.theta = theta # vector of values at states in S

    def __call__(self, s: Any) -> float:
        w = np.array([self.k(s, s_prime) for s_prime in self.S])
        w /= np.linalg.norm(w, ord=1)
        return np.dot(self.theta, w)

    def fit(self, S: list[Any], U: np.ndarray):
        self.theta = U

class MultilinearValueFunction(ApproximateValueFunction):
    def __init__(self, o, delta: np.ndarray, theta: np.ndarray): # TODO - typing
        self.o = o
        self.delta = delta
        self.theta = theta

    def __call__(self, s: Any) -> float: # TODO - Implement
        pass

    def fit(self, S: list[Any], U: np.ndarray):
        self.theta = U

class SimplexValueFunction(ApproximateValueFunction):
    def __init__(self, o, delta: np.ndarray, theta: np.ndarray): # TODO - typing
        self.o = o
        self.delta = delta
        self.theta = theta

    def __call__(self, s: Any) -> float: # TODO - Implement
        pass

    def fit(self, S: list[Any], U: np.ndarray):
        self.theta = U

class LinearRegressionValueFunction(ApproximateValueFunction):
    def __init__(self, beta: Callable[[Any], np.ndarray], theta: np.ndarray):
        self.beta = beta # basis vector function
        self.theta = theta # vector of parameters

    def __call__(self, s: Any) -> float:
        return np.dot(self.theta, self.beta(s))

    def fit(self, S: list[Any], U: np.ndarray):
        X = np.array([self.beta(s) for s in S])
        self.theta = np.linalg.solve(X.T @ X, X.T @ U)

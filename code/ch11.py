import numpy as np
import random

from abc import ABC, abstractmethod
from typing import Any, Callable

from ch07 import MDP
from convenience import normalize


class PolicyGradientEstimationMethod(ABC):
    @abstractmethod
    def gradient(self, policy: Callable[[np.ndarray, Any], Any], theta: np.ndarray) -> np.ndarray:
        pass


class FiniteDifferenceGradient(PolicyGradientEstimationMethod):
    def __init__(self, P: MDP, b: np.ndarray, d: int, m: int, delta: float):
        self.P = P          # problem
        self.b = b          # initial state distribution
        self.d = d          # depth
        self.m = m          # number of samples
        self.delta = delta  # step size

    def gradient(self, policy: Callable[[np.ndarray, Any], Any], theta: np.ndarray) -> np.ndarray:
        n = len(theta)
        def delta_theta(i): np.array([self.delta if i == k else 0.0 for k in range(n)])
        def R(tau): return np.sum([r*(self.P.gamma**k) for (k, (s, a, r)) in enumerate(tau)])

        def U(theta_prime):  # TODO - Rethink naming conventions
            def pi(s): return policy(theta_prime, s)
            def tau(): return self.P.simulate(random.choices(self.P.S, weights=self.b)[0], pi, self.d)
            return np.mean([R(tau()) for i in range(self.m)])
        grad_U = np.array([(U(theta + delta_theta(i)) - U(theta)) for i in range(n)])
        return grad_U / self.delta


class RegressionGradient(PolicyGradientEstimationMethod):
    def __init__(self, P: MDP, b: np.ndarray, d: int, m: int, delta: float):
        self.P = P          # problem
        self.b = b          # initial state distribution
        self.d = d          # depth
        self.m = m          # number of samples
        self.delta = delta  # step size

    def gradient(self, policy: Callable[[np.ndarray, Any], Any], theta: np.ndarray) -> np.ndarray:
        delta_theta = self.delta * normalize(np.randn(self.m, len(theta)), ord=2, axis=1, keepdims=True)
        def R(tau): return np.sum([r*(self.P.gamma**k) for (k, (s, a, r)) in enumerate(tau)])

        def U(theta_prime):  # TODO - Rethink naming conventions
            def pi(s): return policy(theta_prime, s)
            def tau(): return self.P.simulate(random.choices(self.P.S, weights=self.b)[0], pi, self.d)
            return R(tau())
        grad_U = np.array([(U(theta + row) - U(theta)) for row in delta_theta])
        return np.linalg.solve(delta_theta.T @ delta_theta, delta_theta.T @ grad_U)


class LikelihoodRatioGradient(PolicyGradientEstimationMethod):
    def __init__(self, P: MDP, b: np.ndarray, d: int, m: int, grad_ll: Callable[[np.ndarray, Any, Any], np.ndarray]):
        self.P = P              # problem
        self.b = b              # initial state distribution
        self.d = d              # depth
        self.m = m              # number of samples
        self.grad_ll = grad_ll  # gradient of log likelihood

    def gradient(self, policy: Callable[[np.ndarray, Any], Any], theta: np.ndarray, return_FIM=False) -> np.ndarray:  # TODO - Check inheritance for Python and adding parameter
        def policy_theta(s): return policy(theta, s)
        def R(tau): return np.sum([r*(self.P.gamma**k) for (k, (s, a, r)) in enumerate(tau)])
        def grad_log(tau): return np.sum([self.grad_ll(theta, a, s) for (s, a, r) in tau])
        def grad_U(tau): return grad_log(tau) * R(tau)  # TODO - Maybe there is a bug in the textbook (I found two others - need to go back and find them)
        def traj(): return self.P.simulate(random.choices(self.P.S, weights=self.b)[0], policy_theta, self.d)
        trajs = [traj() for _ in range(self.m)]
        avg_grad = np.mean([grad_U(tau) for tau in trajs])  # TODO - Check the dimension of this answer
        if return_FIM:
            def F(tau): return np.outer(grad_log(tau), grad_log(tau))
            return avg_grad, np.mean([F(tau) for tau in trajs])
        return avg_grad


def RewardToGoGradient(PolicyGradientEstimationMethod):
    def __init__(self, P: MDP, b: np.ndarray, d: int, m: int, grad_ll: Callable[[np.ndarray, Any, Any], np.ndarray]):
        self.P = P              # problem
        self.b = b              # initial state distribution
        self.d = d              # depth
        self.m = m              # number of samples
        self.grad_ll = grad_ll  # gradient of log likelihood

    def gradient(self, policy: Callable[[np.ndarray, Any], Any], theta: np.ndarray) -> np.ndarray:
        def policy_theta(s): return policy(theta, s)
        def R(tau, j): return np.sum([r*(self.P.gamma**k) for (k, (s, a, r)) in zip(range(j, self.d + 1), tau[j:])])  # TODO - Note, this might be a bug, if range(j, self.d + 1) and tau[j:] are not same length
        def grad_U(tau): return np.sum([self.grad_ll(theta, a, s) * R(tau, j) for (j, (s, a, r)) in enumerate(tau)])
        def traj(): return self.P.simulate(random.choices(self.P.S, weights=self.b)[0], policy_theta, self.d)
        return np.mean([grad_U(traj()) for _ in range(self.m)])  # TODO - Check the dimension of this answer


def BaselineSubtractionGradient(PolicyGradientEstimationMethod):
    def __init__(self, P: MDP, b: np.ndarray, d: int, m: int, grad_ll: Callable[[np.ndarray, Any, Any], np.ndarray]):
        self.P = P              # problem
        self.b = b              # initial state distribution
        self.d = d              # depth
        self.m = m              # number of samples
        self.grad_ll = grad_ll  # gradient of log likelihood

    def gradient(self, policy: Callable[[np.ndarray, Any], Any], theta: np.ndarray) -> np.ndarray:
        def policy_theta(s): return policy(theta, s)
        def ell(a, s, k): return self.grad_ll(theta, a, s)*(self.P.gamma**k)
        def R(tau, k): return np.sum([r*(self.P.gamma**j) for (j, (s, a, r)) in enumerate(tau[k:])])
        def numer(tau): return np.sum([(ell(a, s, k)**2)*R(tau, k) for (k, (s, a, r)) in enumerate(tau)])
        def denom(tau): return np.sum([ell(a, s, k)**2 for (k, (s, a, r)) in enumerate(tau)])  # TODO - Maybe another bug in Julia?
        def base(tau): return np.divide(numer(tau), denom(tau))
        def traj(): return self.P.simulate(random.choices(self.P.S, weights=self.b)[0], policy_theta, self.d)
        trajs = [traj() for _ in range(self.m)]
        rbase = np.mean([base(tau) for tau in trajs])
        def grad_U(tau): return np.sum([(ell(a, s, k) * (R(tau, k) - rbase)) for (k, (s, a, r)) in enumerate(tau)])
        return np.mean([grad_U(tau) for tau in trajs])  # TODO - Check the dimension of this answer

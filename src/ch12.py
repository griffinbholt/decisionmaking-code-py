import numpy as np
import random

from abc import ABC, abstractmethod
from typing import Any, Callable

from ch07 import MDP
from ch11 import PolicyGradientEstimationMethod, LikelihoodRatioGradient


class ParametrizedPolicyUpdate(ABC):
    @abstractmethod
    def update(self, theta: np.ndarray) -> np.ndarray:
        pass


class PolicyGradientEstimate():
    def __init__(self, policy: Callable[[np.ndarray, Any], Any], M: PolicyGradientEstimationMethod):
        self.policy = policy
        self.M = M

    def __call__(self, theta: np.ndarray, return_FIM=False):
        if return_FIM:
            return self.M.gradient(self.policy, theta, return_FIM=True)
        return self.M.gradient(self.policy, theta)


class PolicyGradientUpdate(ParametrizedPolicyUpdate):
    def __init__(self, grad_U: PolicyGradientEstimate, alpha: float):
        self.grad_U = grad_U  # policy gradient estimate
        self.alpha = alpha    # step factor

    def update(self, theta: np.ndarray):
        return theta + (self.alpha * self.grad_U(theta))


def scale_gradient(grad: np.ndarray, l2_max: float) -> np.ndarray:
    return min(l2_max / np.linalg.norm(grad), 1) * grad


def clip_gradient(grad: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip(grad, a, b)


# TODO - Notify that I have totally changed this class from Julia to be simpler
class RestrictedPolicyUpdate(ParametrizedPolicyUpdate):
    def __init__(self, grad_U: PolicyGradientEstimate, epsilon: float):
        self.grad_U = grad_U    # policy gradient estimate
        self.epsilon = epsilon  # divergence bound

    def update(self, theta: np.ndarray):
        u = self.grad_U(theta)
        return theta + (u * np.sqrt((2*self.epsilon) / np.dot(u, u)))


class NaturalPolicyUpdate(ParametrizedPolicyUpdate):
    def __init__(self, grad_U: PolicyGradientEstimate, epsilon: float):
        assert type(self.grad_U.M) == LikelihoodRatioGradient  # TODO - Check if works
        self.grad_U = grad_U    # policy gradient estimate (specifically, likelihood ratio gradient estimate)
        self.epsilon = epsilon  # divergence bound

    def update(self, theta: np.ndarray, return_trajs=False):  # TODO - Function return signature
        del_U, F, trajs = self.grad_U(theta, return_FIM=True, return_trajs=return_trajs)
        u = np.linalg.solve(F, del_U)
        theta_prime = theta + (u * np.sqrt((2*self.epsilon) / np.dot(del_U, u)))
        if return_trajs:
            return theta_prime, trajs
        return theta_prime


class TrustRegionUpdate(NaturalPolicyUpdate):
    def __init__(self,
                 grad_U: PolicyGradientEstimate,
                 p: Callable[[np.ndarray, Any, Any], float],
                 KL: Callable[[np.ndarray, np.ndarray, Any], float],
                 epsilon: float,
                 alpha: float):
        super().__init__(grad_U, epsilon)
        self.p = p          # policy likelihood p(theta, a, s)
        self.KL = KL        # KL divergence KL(theta, theta_prime, s)
        self.alpha = alpha  # line search reduction factor (e.g., 0.5)

    def surrogate_objective(self,
                            theta: np.ndarray,
                            theta_prime: np.ndarray,
                            trajs: list[tuple[Any, Any, float]]) -> float:
        d, p, gamma = self.grad_U.M.d, self.p, self.grad_U.M.P.gamma
        def R(tau, j): return np.sum([r * (gamma**k) for (k, (s, a, r)) in zip(range(j, d + 1), tau[j:])])
        def w(a, s): return p(theta_prime, a, s) / p(theta, a, s)
        def f(tau): np.mean([w(a, s) * R(tau, k) for (k, (s, a, r)) in enumerate(tau)])
        return np.mean([f(tau) for tau in trajs])

    def surrogate_constraint(self,
                            theta: np.ndarray,
                            theta_prime: np.ndarray,
                            trajs: list[tuple[Any, Any, float]]) -> float:
        gamma = self.grad_U.M.P.gamma
        def KL(tau): return np.mean([self.KL(theta, theta_prime, s) * (gamma**k) for (k, (s, a, r)) in enumerate(tau)])
        return np.mean([KL(tau) for tau in trajs])

    def linesearch(self, 
                   f: Callable[[np.ndarray], float],
                   g: Callable[[np.ndarray], float],
                   theta: np.ndarray,
                   theta_prime: np.ndarray):
        ftheta = f(theta)
        while (g(theta_prime) > self.epsilon) or (f(theta_prime) <= ftheta):
            theta_prime = theta + self.alpha * (theta_prime - theta)
        return theta_prime

    def update(self, theta: np.ndarray) -> np.ndarray:
        theta_prime, trajs = super().update(theta, return_trajs=True)  # Natural Policy Update
        def f(theta_prime): return self.surrogate_objective(theta, theta_prime, trajs)
        def g(theta_prime): return self.surrogate_constraint(theta, theta_prime, trajs)
        return self.linesearch(f, g, theta, theta_prime)


class ClampedSurrogateUpdate(ParametrizedPolicyUpdate):
    def __init__(self,
                 P: MDP,
                 b: np.ndarray,
                 d: int,
                 m: int,
                 policy: Callable[[np.ndarray, Any], Any],
                 p: Callable[[np.ndarray, Any, Any], float],
                 grad_p: Callable[[np.ndarray, Any, Any], np.ndarray],
                 epsilon: float,
                 alpha: float,
                 k_max: int):
        self.P = P              # problem
        self.b = b              # initial state distribution
        self.d = d              # depth
        self.m = m              # number of trajectories
        self.policy = policy    # policy
        self.p = p              # policy likelihood
        self.grad_p = grad_p    # policy likelihood gradient
        self.epsilon = epsilon  # divergence bound
        self.alpha = alpha      # step size
        self.k_max = k_max      # number of iterations per update
    
    def clamped_gradient(self, theta: np.ndarray, theta_prime: np.ndarray, trajs: list[tuple[Any, Any, float]]) -> np.ndarray:
        def R(tau, j): return np.sum([r*(self.P.gamma**k) for (k, (s, a, r)) in zip(range(j, self.d + 1), tau[j:])])
        def grad_f(a, s, r_togo):
            P = self.p(theta, a, s)
            w = self.p(theta_prime, a, s)
            if ((r_togo > 0) and (w > 1 + self.epsilon)) or ((r_togo < 0) and (w < 1 - self.epsilon)):
                return np.zeros(len(theta))
            return self.grad_p(theta_prime, a, s) * (r_togo / P)
        def grad_f_traj(tau): return np.mean([grad_f(a, s, R(tau, k)) for (k, (s, a, r)) in enumerate(tau)])
        return np.mean([grad_f_traj(tau) for tau in trajs])

    def update(self, theta: np.ndarray) -> np.ndarray:
        def policy_theta(s): return self.policy(theta, s)
        def traj(): return self.P.simulate(random.choices(self.P.S, weights=self.b)[0], policy_theta, self.d)
        trajs = [traj() for _ in range(self.m)]
        theta_prime = theta.copy()
        for _ in range(self.k_max):
            theta_prime += self.alpha * self.clamped_gradient(theta, theta_prime, trajs)
        return theta_prime

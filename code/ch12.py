import numpy as np

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
        self.grad_U = grad_U # policy gradient estimate 
        self.alpha = alpha # step factor

    def update(self, theta: np.ndarray):
        return theta + (self.alpha * self.grad_U(theta))

def scale_gradient(grad: np.ndarray, l2_max: float) -> np.ndarray:
    return min(l2_max / np.linalg.norm(grad), 1) * grad

def clip_gradient(grad: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip(grad, a, b)

# TODO - Notify that I have totally changed this class from Julia to be simpler
class RestrictedPolicyUpdate(ParametrizedPolicyUpdate):
    def __init__(self, grad_U: PolicyGradientEstimate, epsilon: float):
        self.grad_U = grad_U # policy gradient estimate
        self.epsilon = epsilon # divergence bound

    def update(self, theta: np.ndarray):
        u = self.grad_U(theta)
        return theta + (u * np.sqrt((2*self.epsilon) / np.dot(u, u)))

class NaturalPolicyUpdate(ParametrizedPolicyUpdate):
    def __init__(self, grad_U: PolicyGradientEstimate, epsilon: float):
        assert type(self.grad_U.M) == LikelihoodRatioGradient # TODO - Check if works
        self.grad_U = grad_U # policy gradient estimate (specifically, likelihood ratio gradient estimate)
        self.epsilon = epsilon # divergence bound

    def update(self, theta: np.ndarray):
        del_U, F = self.grad_U(theta, return_FIM=True)
        u = np.linalg.solve(F, del_U)
        return theta + (u * np.sqrt((2*self.epsilon) / np.dot(del_U, u)))

class TrustRegionUpdate(ParametrizedPolicyUpdate):
    pass # TODO

class ClampedSurrogateUpdate(ParametrizedPolicyUpdate):
    pass # TODO

import numpy as np

from abc import abstractmethod
from typing import Any, Callable, Type
from scipy.stats import rv_continuous, multivariate_normal

from ch07 import MDP, SolutionMethod
from ch09 import rollout

class MonteCarloPolicyEvaluation():
    def __init__(self, P: MDP, b: np.ndarray, d: int, m: int):
        self.P = P # problem
        self.b = b # initial state distribution
        self.d = d # depth
        self.m = m # number of samples

    def evaluate_policy(self, policy: Callable[[Any], Any]):
        state = np.random.choice(self.P.S, p=self.b)
        return np.mean([rollout(self.P, state, policy, self.d) for _ in range(self.m)])

    def evaluate_parametrized_policy(self, policy: Callable[[np.ndarray, Any], Any], theta: np.ndarray):
        return self.evaluate_policy(lambda s: policy(theta, s))

class PolicySearchMethod(SolutionMethod):
    @abstractmethod
    def optimize(self, policy: Callable[[np.ndarray, Any], Any], U: MonteCarloPolicyEvaluation) -> np.ndarray:
        pass

class SearchDistributionMethod(SolutionMethod):
    @abstractmethod
    def optimize_dist(self, policy: Callable[[np.ndarray, Any], Any], U: MonteCarloPolicyEvaluation) -> rv_continuous:
        pass

class HookeJeevesPolicySearch(PolicySearchMethod):
    def __init__(self, theta: np.ndarray, alpha: float, c: float, epsilon: float):
        theta = self.theta # initial parameterization
        alpha = self.alpha # step size
        c = self.c # step size reduction factor
        epsilon = self.epsilon # termination step size

    def optimize(self, policy: Callable[[np.ndarray, Any], Any], U: MonteCarloPolicyEvaluation) -> np.ndarray:
        theta, theta_prime = self.theta.copy(), np.zeros(self.theta.shape)
        u, n = U(policy, theta), len(theta)
        while self.alpha > self.epsilon:
            np.copyto(dst=theta_prime, src=theta)
            best = (0, 0, u) # (i, sgn, u)
            for i in range(n):
                for sgn in [-1, 1]:
                    theta_prime[i] = theta[i] + sgn*self.alpha 
                    u_prime = U(policy, theta_prime)
                    if u_prime > best[2]:
                        best = (i, sgn, u_prime)
                theta_prime[i] = theta[i]
            if best[0] != 0:
                theta[best[0]] += best[1]*self.alpha
                u = best[2]
            else:
                self.alpha *= self.c
        return theta

class GeneticPolicySearch(PolicySearchMethod):
    def __init__(self, thetas: list[np.ndarray], sigma: float, m_elite: int, k_max: int):
        self.thetas = thetas # initial population
        self.sigma = sigma # initial standard deviation
        self.m_elite = m_elite # number of elite samples
        self.k_max = k_max # number of iterations

    def optimize(self, policy: Callable[[np.ndarray, Any], Any], U: MonteCarloPolicyEvaluation) -> np.ndarray:
        thetas = self.thetas.copy()
        n, m = len(thetas[0]), len(thetas)
        for k in range(self.k_max):
            us = np.array([U(policy, theta) for theta in thetas])
            sp = np.flip(np.argsort(us))
            theta_best = thetas[sp[0]]
            rand_elite = lambda: thetas[sp[np.random.randint(low=0, high=self.m_elite)]]
            thetas = [rand_elite() + self.sigma * np.random.randn(n) for _ in range(m - 1)]
            thetas.append(theta_best)
        return thetas[-1]

"""
Due to the limitations of the scipy.stats library (distributions lack a `fit` function), 
we implement this for the Multivariate Gaussian distribution specifically
(instead of generically, for any multivariable distribution).

However, this class can easily be changed to work for any distribution provided you know how to fit that distribution
"""
class CrossEntropyPolicySearch(SearchDistributionMethod, PolicySearchMethod):
    def __init__(self, p: multivariate_normal, m: int, m_elite: int, k_max: int):
        self.p = p # initial distribution
        self.m = m # number of samples
        self.m_elite = m_elite # number of elite samples
        self.k_max = k_max # number of iterations

    def optimize_dist(self, policy: Callable[[np.ndarray, Any], Any], U: MonteCarloPolicyEvaluation) -> multivariate_normal:
        p = self.p 
        for _ in range(self.k_max):
            thetas = p.rvs(self.m)
            us = np.array([U(policy, theta) for theta in thetas])
            theta_elite = thetas[np.argsort(us)[(self.m - self.m_elite):]]
            p = multivariate_normal(np.mean(theta_elite, axis=0), np.cov(theta_elite, rowvar=0))
        return p

    def optimize(self, policy: Callable[[np.ndarray, Any], Any], U: MonteCarloPolicyEvaluation) -> np.ndarray:
        return self.optimize_dist(policy, U).mean

class EvolutionStrategies(SearchDistributionMethod):
    pass
    # def __init__(self, D: Type[rv_continuous], psi: )

class IsotropicEvolutionStrategies(EvolutionStrategies):
    pass
    
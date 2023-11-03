import numpy as np
import random

from abc import abstractmethod
from typing import Any, Callable, Type
from scipy.stats import rv_continuous, multivariate_normal

from ch07 import MDP, SolutionMethod
from ch09 import rollout


class MonteCarloPolicyEvaluation():
    def __init__(self, P: MDP, b: np.ndarray, d: int, m: int):
        self.P = P  # problem
        self.b = b  # initial state distribution
        self.d = d  # depth
        self.m = m  # number of samples

    def evaluate_policy(self, policy: Callable[[Any], Any]) -> float:
        state = random.choices(self.P.S, weights=self.b)[0]
        return np.mean([rollout(self.P, state, policy, self.d) for _ in range(self.m)])

    def evaluate_parametrized_policy(self, policy: Callable[[np.ndarray, Any], Any], theta: np.ndarray) -> float:
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
        self.theta = theta      # initial parameterization
        self.alpha = alpha      # step size
        self.c = c              # step size reduction factor
        self.epsilon = epsilon  # termination step size

    def optimize(self, policy: Callable[[np.ndarray, Any], Any], U: MonteCarloPolicyEvaluation) -> np.ndarray:
        theta, theta_prime = self.theta.copy(), np.zeros(self.theta.shape)
        u, n = U(policy, theta), len(theta)
        while self.alpha > self.epsilon:
            np.copyto(dst=theta_prime, src=theta)
            best = (0, 0, u)  # (i, sgn, u)
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
        self.thetas = thetas    # initial population
        self.sigma = sigma      # initial standard deviation
        self.m_elite = m_elite  # number of elite samples
        self.k_max = k_max      # number of iterations

    def optimize(self, policy: Callable[[np.ndarray, Any], Any], U: MonteCarloPolicyEvaluation) -> np.ndarray:
        thetas = self.thetas.copy()
        n, m = len(thetas[0]), len(thetas)
        for k in range(self.k_max):
            us = np.array([U(policy, theta) for theta in thetas])
            sp = np.flip(np.argsort(us))
            theta_best = thetas[sp[0]]
            def rand_elite(): thetas[sp[np.random.randint(low=0, high=self.m_elite)]]
            thetas = [rand_elite() + self.sigma * np.random.randn(n) for _ in range(m - 1)]
            thetas.append(theta_best)
        return thetas[-1]


class CrossEntropyPolicySearch(SearchDistributionMethod, PolicySearchMethod):
    """
    Due to the limitations of the scipy.stats library (distributions lack a `fit` function),
    we implement this for the Multivariate Gaussian distribution specifically
    (instead of generically, for any multivariable distribution).

    However, this class can easily be changed to work for any distribution provided you know how to fit that distribution
    """
    def __init__(self, p: multivariate_normal, m: int, m_elite: int, k_max: int):
        self.p = p              # initial distribution
        self.m = m              # number of samples
        self.m_elite = m_elite  # number of elite samples
        self.k_max = k_max      # number of iterations

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


def evolution_strategy_weights(m: int) -> np.ndarray:
    ws = np.array([max(0, np.log((m/2) + 1) - np.log(i + 1)) for i in range(m)])
    ws /= np.sum(ws)
    ws -= (1/m)
    return ws


class EvolutionStrategies(SearchDistributionMethod):
    """
    Due to the limitations of the scipy.stats library (distributions aren't easily generalized as in Julia),
    we implement this for the Multivariate Gaussian distribution specifically
    (instead of generically, for any multivariable distribution).

    However, this class can easily be changed to work for any distribution you wish.
    """
    def __init__(self, mu: np.ndarray, Sigma: np.ndarray, m: int, alpha: float, k_max: int):
        self.mu = mu                     # initial mean 
        self.A = self._decompose(Sigma)  # initial A, where A.T @ A = Sigma (the covariance matrix)
        self.m = m                       # number of samples
        self.alpha = alpha               # step factor
        self.k_max = k_max               # number of iterations
    
    def _decompose(self, Sigma: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eig(Sigma)
        return np.diag(np.sqrt(eigvals)) @ eigvecs.T
    
    def _grad_ll(self, mu: np.ndarray, A: np.ndarray, x: np.ndarray) -> [np.ndarray, np.ndarray]:
        """log search likelihood gradient for the Multivariable Normal distribution"""
        Sigma = A.T @ A
        diff = x - mu
        inv_Sigma = np.linalg.inv(Sigma)
        grad_ll_mu = inv_Sigma @ diff
        grad_ll_Sigma = 0.5 * (np.outer(inv_Sigma @ diff, inv_Sigma.T @ diff) - inv_Sigma)
        grad_ll_A = A @ (grad_ll_Sigma + grad_ll_Sigma.T)
        return grad_ll_mu, grad_ll_A

    def optimize_dist(self, policy: Callable[[np.ndarray, Any], Any], U: MonteCarloPolicyEvaluation) -> multivariate_normal:
        mu, A = self.mu, self.A
        ws = evolution_strategy_weights(self.m)
        for _ in range(self.k_max):
            thetas = multivariate_normal.rvs(mean=mu, cov=(A.T @ A), size=self.m)
            us = np.array([U(policy, thetas[:, i]) for i in range(self.m)])
            sp = np.flip(np.argsort(us))
            update_mu, update_A = np.zeros_like(mu), np.zeros_like(A)
            for (w, i) in zip(ws, sp):  # Note: Accomplished in Julia in one line
                grad_ll_mu, grad_ll_A = self._grad_ll(mu, A, thetas[:, i])
                update_mu += w * grad_ll_mu
                update_A += w * grad_ll_A
            mu += self.alpha * update_mu
            A += self.alpha * update_A
        return multivariate_normal(mean=mu, cov=(A.T @ A))


class IsotropicEvolutionStrategies(SearchDistributionMethod):
    """
    An evolution strategies method for updating an isotropic multivariate
    Gaussian search distribution with mean `mu` and covariance (sigma^2) * I
    over policy parameterizations for a policy `policy(theta, s)`.

    This implementation also takes a policy evaluation function `U`, a step
    factor `alpha`, and an iteration count `k_max`.

    In each iteration, m/2 parametrization samples are drawn and mirrored and
    are then used to estimate the search gradient.
    """
    def __init__(self, mu: np.ndarray, sigma: float, m: int, alpha: float, k_max: int):
        self.mu = mu                     # initial mean
        self.sigma = sigma               # initial standard deviation
        self.m = m                       # number of samples
        self.alpha = alpha               # step factor
        self.k_max = k_max               # number of iterations
    
    def optimize_dist(self, policy: Callable[[np.ndarray, Any], Any], U: MonteCarloPolicyEvaluation) -> multivariate_normal:
        mu, sigma = self.mu, self.sigma
        n = len(self.p)
        ws = evolution_strategy_weights(2 * (self.m // 2))
        for _ in range(self.k_max):
            epsilons = np.random.randn(self.m // 2, n)
            epsilons = np.append(epsilons, -epsilons)
            us = np.array([U(policy, mu + sigma * eps) for eps in epsilons])
            sp = np.flip(np.argsort(us))
            update = np.sum([w * epsilons[i] for (w, i) in zip(ws, sp)]) / sigma
            mu += self.alpha * update
        return multivariate_normal(mean=mu, cov=sigma)

"""Chapter 13: Actor-Critic Methods"""

import numpy as np
import random

from abc import ABC, abstractmethod
from typing import Any, Callable

from ch07 import MDP


class ActorCriticMethod(ABC):
    def __init__(self, P: MDP, b: np.ndarray, d: int, m: int):
        self.P = P  # problem
        self.b = b  # initial state distribution
        self.d = d  # depth
        self.m = m  # number of samples

    @abstractmethod
    def gradient(self,
                 policy: Callable[[np.ndarray, Any], Any],
                 theta: np.ndarray, 
                 phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass


class BasicActorCritic(ActorCriticMethod):
    def __init__(self, 
                 P: MDP,
                 b: np.ndarray,
                 d: int,
                 m: int,
                 grad_ll: Callable[[np.ndarray, Any, Any], np.ndarray],
                 U: Callable[[np.ndarray, Any], float],
                 grad_U: Callable[[np.ndarray, Any], np.ndarray]):
        super().__init__(P, b, d, m)
        self.grad_ll = grad_ll  # gradient of log likelihood grad_ll(theta, a, s)
        self.U = U              # parametrized value function U(theta, s)
        self.grad_U = grad_U    # gradient of value function grad_U(theta, s)
    
    def gradient(self, policy: Callable[[np.ndarray, Any], Any], theta: np.ndarray, phi: np.ndarray):
        gamma = self.P.gamma
        def policy_theta(s): return policy(theta, s)
        def R(tau, j): return np.sum([r*(gamma**k) for (k, (s, a, r)) in enumerate(tau[j:])])
        def A(tau, j): return tau[j][2] + gamma * self.U(phi, tau[j + 1][0]) - self.U(phi, tau[j][0])
        def grad_U_theta(tau): return np.sum([self.grad_ll(theta, a, s) * A(tau, j) * (gamma**j) for (j, (s, a, r)) in enumerate(tau[:-1])])
        def grad_ell_theta(tau): np.sum([(self.U(phi, s) - R(tau, j)) * self.grad_U(phi, s) for (j, (s, a, r)) in enumerate(tau)])
        def traj(): return self.P.simulate(random.choices(self.P.S, weights=self.b)[0], policy_theta, self.d)
        trajs = [traj() for _ in range(self.m)]
        return np.mean([grad_U_theta(tau) for tau in trajs]), np.mean([grad_ell_theta(tau) for tau in trajs])


class GeneralizedAdvantageEstimation(BasicActorCritic):
    def __init__(self, 
                 P: MDP,
                 b: np.ndarray,
                 d: int,
                 m: int,
                 grad_ll: Callable[[np.ndarray, Any, Any], np.ndarray],
                 U: Callable[[np.ndarray, Any], float],
                 grad_U: Callable[[np.ndarray, Any], np.ndarray],
                 lam: float):
        super().__init__(P, b, d, m, grad_ll, U, grad_U)
        self.lam = lam  # weight in [0, 1]
    
    def gradient(self, policy: Callable[[np.ndarray, Any], Any], theta: np.ndarray, phi: np.ndarray):
        gamma, lam = self.P.gamma, self.lam
        def policy_theta(s): return policy(theta, s)
        def R(tau, j): return np.sum([r*(gamma**k) for (k, (s, a, r)) in enumerate(tau[j:])])
        def delta(tau, j): return tau[j][2] + gamma * self.U(phi, tau[j + 1][0]) - self.U(phi, tau[j][0])
        def A(tau, j): np.sum([((gamma*lam)**l) * delta(tau, j + l) for l in range(self.d - j)])
        def grad_U_theta(tau): return np.sum([self.grad_ll(theta, a, s) * A(tau, j) * (gamma**j) for (j, (s, a, r)) in enumerate(tau[:-1])])
        def grad_ell_theta(tau): np.sum([(self.U(phi, s) - R(tau, j)) * self.grad_U(phi, s) for (j, (s, a, r)) in enumerate(tau)])
        def traj(): return self.P.simulate(random.choices(self.P.S, weights=self.b)[0], policy_theta, self.d)
        trajs = [traj() for _ in range(self.m)]
        return np.mean([grad_U_theta(tau) for tau in trajs]), np.mean([grad_ell_theta(tau) for tau in trajs])


class DeterministicPolicyGradient(ActorCriticMethod):
    def __init__(self, 
                 P: MDP,
                 b: np.ndarray,
                 d: int,
                 m: int,
                 grad_p: Callable[[np.ndarray, Any], np.ndarray],
                 Q: Callable[[np.ndarray, Any, np.ndarray], float],
                 grad_Q_phi: Callable[[np.ndarray, Any, np.ndarray], np.ndarray],
                 grad_Q_a: Callable[[np.ndarray, Any, np.ndarray], np.ndarray],
                 sigma: float):
        super().__init__(P, b, d, m)
        self.grad_p = grad_p          # gradient of deterministic policy pi(theta, s)
        self.Q = Q                    # parametrized value function Q(phi, s, a)
        self.grad_Q_phi = grad_Q_phi  # gradient of value function with respect to phi
        self.grad_Q_a = grad_Q_a      # gradient of value function with respect to a
        self.sigma = sigma            # policy noise
    
    def gradient(self, policy: Callable[[np.ndarray, Any], np.ndarray], theta: np.ndarray, phi: np.ndarray):  # TODO - lots of questions for Mykel
        gamma = self.P.gamma
        def random_policy(s): return policy(theta, s) + self.sigma  # TODO - Fix this function - not sure what the Julia is saying
        def grad_U_theta(tau): return np.sum([(self.grad_p(theta, s) @ self.grad_Q_a(phi, s, policy(theta, s))) * (gamma**j) for (j, (s, a, r)) in enumerate(tau)])
        def grad_ell_phi_j(tau, j):
            s, a, r = tau[j]
            s_prime = tau[j + 1][0]
            a_prime = policy(theta, s_prime)
            delta = r + gamma * self.Q(phi, s_prime, a_prime) - self.Q(phi, s, a)
            return delta * (gamma * self.grad_Q_phi(phi, s_prime, a_prime) - self.grad_Q_phi(phi, s, a))
        def grad_ell_phi(tau): np.sum([grad_ell_phi_j(tau, j) for j in range(len(tau) - 1)])
        def traj(): return self.P.simulate(random.choices(self.P.S, weights=self.b)[0], random_policy, self.d)
        trajs = [traj() for _ in range(self.m)]
        return np.mean([grad_U_theta(tau) for tau in trajs]), np.mean([grad_ell_phi(tau) for tau in trajs])

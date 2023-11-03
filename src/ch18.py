import cvxpy as cp
import numpy as np
import random

from abc import ABC, abstractmethod
from typing import Callable

from ch07 import MDP
from convenience import normalize


class ImitationLearning(ABC):
    @abstractmethod
    def optimize(self, **kwargs):
        pass


class BehavioralCloning(ImitationLearning):
    def __init__(self, alpha: float, k_max: int, grad_ll: Callable[[np.ndarray, int, int], float]):
        self.alpha = alpha      # step size
        self.k_max = k_max      # number of iterations
        self.grad_ll = grad_ll  # log likelihood gradient - function of (theta, a, s)

    def optimize(self, D: list[tuple[int, int]], theta: np.ndarray) -> np.ndarray:
        for _ in range(self.k_max):
            gradient = np.mean([self.grad_ll(theta, a, s) for (s, a) in D])
            theta += self.alpha * gradient
        return theta


class DataSetAggregation(ImitationLearning):
    def __init__(self,
                 P: MDP,
                 bc: BehavioralCloning,
                 k_max: int,
                 m: int,
                 d: int,
                 b: np.ndarray,
                 expert_policy: Callable[[int], int],
                 param_policy: Callable[[np.ndarray, int], int]):
        self.P = P                          # problem with unknown reward function
        self.bc = bc                        # behavioral cloning instance
        self.k_max = k_max                  # number of iterations
        self.m = m                          # number of rollouts per iteration
        self.d = d                          # rollout depths
        self.b = b                          # initial state distribution
        self.expert_policy = expert_policy  # expert
        self.param_policy = param_policy    # parametrized policy

    def optimize(self, D: list[tuple[int, int]], theta: np.ndarray) -> np.ndarray:
        theta = self.bc.optimize(D, theta)
        for _ in range(self.k_max - 1):
            for _ in range(self.m):
                s = random.choices(self.P.S, weights=self.b)[0]  # TODO - Check with Mykel & Tim about this - what form is the state distribution?
                for _ in range(self.d):
                    D.append((s, self.expert_policy(s)))
                    a = random.choices(self.P.A, weights=self.param_policy(theta, s))[0]  # TODO - Check with Mykel & Tim about this structure
                    s = random.choices(self.P.S, weights=[self.P.T(s, a, s_prime) for s_prime in self.P.S])[0]
            theta = self.bc.optimize(D, theta)
        return theta


class SMILe(ImitationLearning):
    def __init__(self,
                 P: MDP,
                 bc: BehavioralCloning,
                 k_max: int,
                 m: int,
                 d: int,
                 b: np.ndarray,
                 beta: float,
                 expert_policy: Callable[[int], int],
                 param_policy: Callable[[np.ndarray, int], int]):
        self.P = P                          # problem with unknown reward function
        self.bc = bc                        # behavioral cloning instance
        self.k_max = k_max                  # number of iterations
        self.m = m                          # number of rollouts per iteration
        self.d = d                          # rollout depths
        self.b = b                          # initial state distribution
        self.beta = beta                    # mixing scalar (e.g., d^-3)
        self.expert_policy = expert_policy  # expert
        self.param_policy = param_policy    # parametrized policy

    def optimize(self, theta: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        thetas = []
        def policy(s): return self.expert_policy(s)
        for k in range(self.k_max):
            # Execute latest policy to get new data set D
            D = []
            for _ in range(self.m):
                s = random.choices(self.P.S, weights=self.b)[0]  # TODO - Check with Mykel & Tim about this - what form is the state distribution?
                for _ in range(self.d):
                    D.append((s, self.expert_policy(s)))
                    a = policy(s)
                    s = random.choices(self.P.S, weights=[self.P.T(s, a, s_prime)[0] for s_prime in self.P.S])

            # Train the new policy classifier
            theta = self.bc.optimize(D, theta)
            thetas.append(theta)

            # Compute a new policy mixture
            P = normalize(np.array([(1 - self.beta)**i for i in range(k)]), ord=1)
            def policy(s: int, k: int = k, P: np.ndarray = P) -> int:
                if np.random.rand() < ((1 - self.beta)**k):
                    return self.expert_policy(s)
                past_theta = thetas[np.random.choice(k, p=P)]
                return random.choices(self.P.A, weights=self.param_policy(past_theta, s))[0]
        probs = normalize(np.array([(1 - self.beta)**i for i in range(self.k_max)]), ord=1)
        return probs, thetas


class InverseReinforcementLearning(ImitationLearning):
    def __init__(self,
                 P: MDP,
                 b: np.ndarray,
                 d: int,
                 m: int,
                 policy: Callable[[np.ndarray, int], int],
                 beta: Callable[[int, int], np.ndarray],
                 mu_E: np.ndarray,
                 RL,  # TODO - The RL method is unclear - What exactly is it?
                 epsilon: float):
        self.P = P              # problem
        self.b = b              # initial state distribution
        self.d = d              # depth
        self.m = m              # number of samples
        self.policy = policy    # parametrized policy
        self.beta = beta        # binary feature mapping
        self.mu_E = mu_E        # expert feature expectations
        self.RL = RL            # reinforcement learning method
        self.epsilon = epsilon  # tolerance

    def optimize(self, theta: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        thetas = [theta]
        mus = np.array([self.feature_expectations(lambda s: self.policy(theta, s))])
        while True:
            t, phi = self.calc_weighting(mus)
            if t <= self.epsilon:
                break
            np.copyto(self.RL.phi, phi)  # R(s, a) = phi.T @ beta(s, a)
            theta = self.RL.optimize(self.policy, theta)  # TODO - Unclear what RL is
            thetas.append(theta)
            mus = np.vstack([mus, [self.feature_expectations(lambda s: self.policy(theta, s))]])
        lam = self.calc_policy_mixture(mus)
        return lam, thetas

    def feature_expectations(self, policy: Callable[[int], int]) -> float:
        gamma = self.P.gamma
        def mu(tau): return np.sum([(gamma**k) * self.beta(s, a) for (k, (s, a)) in enumerate(tau)], axis=1)  # TODO - Check axis
        trajs = [self.P.simulate(random.choices(self.P.S, weights=self.b)[0], policy, self.d) for _ in range(self.m)]
        return np.mean([mu(tau) for tau in trajs], axis=1)  # TODO - Check axis

    def calc_weighting(self, mus: np.ndarray) -> tuple[float, np.ndarray]:
        # mus: each row of mus is mu_i
        t = cp.Variable()
        phi = cp.Variable((len(self.mu_E), 1))
        objective = cp.Maximize(t)
        constraints = [phi >= 0, cp.norm2(phi) <= 1, phi.T @ self.mu_E >= mus @ phi + t]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return t.value, phi.value.T[0]

    def calc_policy_mixture(self, mus: np.ndarray) -> np.ndarray:
        # mus: each row of mus is mu_i
        k = mus.shape[0]
        lam = cp.Variable(k)
        objective = cp.Minimize(cp.norm2(self.mu_E - mus.T @ lam))
        constraints = [lam >= 0, cp.sum(lam) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return lam.value


class MaximumEntropyIRL(ImitationLearning):
    def __init__(self,
                 P: MDP,
                 b: np.ndarray,
                 d: int,
                 policy: Callable[[np.ndarray, int], int],
                 p: Callable[[np.ndarray, int, int], int],
                 grad_R: Callable[[np.ndarray, int, int]],
                 RL,  # TODO - The RL method is unclear - What exactly is it?
                 alpha: float,
                 k_max: int):
        self.P = P            # problem
        self.b = b            # initial state distribution
        self.d = d            # depth
        self.policy = policy  # parametrized policy
        self.p = p            # parametrized policy likelihood p(theta, a, s)
        self.grad_R = grad_R  # reward function gradient
        self.RL = RL          # reinforcement learning method
        self.alpha = alpha    # step size
        self.k_max = k_max    # number of iterations

    def optimize(self, D: list[tuple[int, int]], phi: np.ndarray, theta: np.ndarrau) -> tuple[np.ndarray, np.ndarray]:
        P, p, grad_R = self.P, self.p, self.grad_R
        S, A, gamma, nD = P.S, P.A, P.gamma, len(D)
        for _ in range(self.k_max):
            np.copyto(self.RL.phi, phi)  # R(s, a) = phi.T @ beta(s, a)
            theta = self.RL.optimize(self.policy, theta)  # TODO - Unclear what RL is
            b = self.discounted_state_visitations(theta)
            def grad_R_tau(tau): return np.sum([(gamma**i) * grad_R(phi, s, a) for (i, (s, a, r)) in enumerate(tau)])
            grad_f = np.sum([grad_R_tau(tau) for tau in D]) - nD * np.sum([b[si] * np.sum([p(theta, a, s) * grad_R(phi, s, a) for a in A]) for (si, s) in enumerate(S)])  # TODO - Typo in textbook? What is `ai` used for?
            phi += self.alpha * grad_f
        return phi, theta

    def discounted_state_visitations(self, theta: np.ndarray):
        P, b, d, p = self.P, self.b, self.d, self.p
        S, A, T, gamma = P.S, P.A, P.T, P.gamma
        b_sk = np.zeros(len(S), d)
        # b_sk[:, 0] = [pdf(b, s) for s in S]  # TODO - What is pdf(b, s)? And what are b? (Discrete distribution? Other?)
        for k in range(1, d):
            for (si_prime, s_prime) in enumerate(S):
                b_sk[si_prime, k] = gamma * np.sum([np.sum([b_sk[si, k] * p(theta, a, s) * T(s, a, s_prime) for (si, s) in enumerate(S)]) for a in A])
        return normalize(np.mean(b_sk, axis=1), ord=1)

import numpy as np
import random

from abc import abstractmethod

from ch19 import POMDP
from ch20 import POMDPPolicy, AlphaVectorPolicy, LookaheadAlphaVectorPolicy, POMDPSolutionMethod


class OfflinePlanningMethod(POMDPSolutionMethod):
    def __init__(self, k_max: int):
        self.k_max = k_max  # maximum number of iterations - TODO Maybe move to POMDPSolutionMethod?

    @abstractmethod
    def update(self, P: POMDP, V: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def solve(self, P: POMDP) -> POMDPPolicy:
        pass

    def alphavector_iteration(self, P: POMDP, V: np.ndarray) -> np.ndarray:
        for _ in range(self.k_max):
            V = self.update(P, V)
        return V


class QMDP(OfflinePlanningMethod):
    def __init__(self, k_max: int):
        super().__init__(k_max)

    def update(self, P: POMDP, V: np.ndarray) -> np.ndarray:
        S, A, R, T, gamma = P.S, P.A, P.R, P.T, P.gamma
        V_prime = np.zeros((len(A), len(S)))
        for a in A:
            for s in S:
                exp_rewards = np.sum([T(s, a, s_prime) * np.max(alpha[j] for alpha in V) for (j, s_prime) in enumerate(S)])
                V_prime[a, s] = R(s, a) + gamma * exp_rewards
        return V_prime
    
    def solve(self, P: POMDP) -> AlphaVectorPolicy:
        V = np.zeros((len(P.A), len(P.S)))
        V = self.alphavector_iteration(P, V)
        return AlphaVectorPolicy(P, V, P.A)


class FastInformedBound(OfflinePlanningMethod):
    def __init__(self, k_max: int):
        super().__init__(k_max)

    def update(self, P: POMDP, V: np.ndarray) -> np.ndarray:
        S, A, O_space, R, T, O, gamma = P.S, P.A, P.O_space, P.R, P.T, P.O, P.gamma
        V_prime = np.zeros((len(A), len(S)))
        for a in A:
            for s in S:
                V_prime[a, s] = R(s, a) + gamma * np.sum([np.max([np.sum([O(a, s_prime, o) * T(s, a, s_prime) * alpha[j] for (j, s_prime) in enumerate(S)]) for alpha in V]) for o in O_space])
        return V_prime
    
    def solve(self, P: POMDP) -> AlphaVectorPolicy:
        V = np.zeros((len(P.A), len(P.S)))
        V = self.alphavector_iteration(P, V)
        return AlphaVectorPolicy(P, V, P.A)


def baws_lowerbound(P: POMDP) -> np.ndarray:
    r = np.max([np.min([P.R(s, a) for s in P.S]) for a in P.A]) / (1 - P.gamma)
    alpha = np.full(shape=len(P.S), fill_value=r)
    return alpha


def blind_lowerbound(P: POMDP, k_max: int) -> np.ndarray:
    S, A, T, R, gamma = P.S, P.A, P.T, P.R, P.gamma
    def Q(s, a, alpha): return R(s, a) + gamma * np.sum([T(s, a, s_prime) * alpha[j] for (j, s_prime) in enumerate(S)])
    V = np.array([baws_lowerbound(P) for a in A])
    for _ in range(k_max):
        V = np.array([[Q(s, a, alpha) for s in S] for (alpha, a) in zip(V, A)])
    return V


class PointBasedValueIteration(OfflinePlanningMethod):
    def __init__(self, B: list[np.ndarray], k_max: int):
        super().__init__(k_max)
        self.B = B  # set of belief points

    def update(self, P: POMDP, V: np.ndarray) -> np.ndarray:
        return np.array([P.backup(V, b) for b in self.B])

    def solve(self, P: POMDP) -> LookaheadAlphaVectorPolicy:
        V = np.full(shape=(len(P.A), len(P.S)), fill_value=baws_lowerbound(P))
        V = self.alphavector_iteration(P, V)
        return LookaheadAlphaVectorPolicy(P, V)


class RandomizedPointBasedValueIteration(PointBasedValueIteration):
    def __init__(self, B: list[np.ndarray], k_max: int):
        super().__init__(B, k_max)

    def update(self, P: POMDP, V: np.ndarray) -> np.ndarray:
        V_prime, B_prime = [], self.B.copy()
        while len(B_prime) != 0:
            b = random.choices(B_prime)
            alpha = V[np.argmax([np.dot(alpha, b) for alpha in V])]
            alpha_prime = P.backup(V, b)
            if np.dot(alpha_prime, b) >= np.dot(alpha, b):
                V_prime.append(alpha_prime)
            else:
                V_prime.append(alpha)
            B_prime = [b for b in B_prime if np.max([np.dot(alpha, b) for alpha in V_prime]) < np.max([np.dot(alpha, b) for alpha in V])]
        return np.array(V_prime)

    def solve(self, P: POMDP) -> LookaheadAlphaVectorPolicy:
        V = np.array([baws_lowerbound(P)])
        V = self.alphavector_iteration(P, V)
        return LookaheadAlphaVectorPolicy(P, V)


def SawtoothPolicy(POMDPPolicy):
    def __init__(self, P: POMDP, V: dict[np.ndarray, float]):
        self.P = P  # POMDP problem
        self.V = V  # dictionary mapping beliefs to utilities

    def utility(self, b: np.ndarray) -> float:
        if b in self.V:
            return V[b]
        n = len(P.s)
        E = np.eye(n)
        u = np.sum([V[E[i]] * b[i] for i in range(n)])
        for (b_prime, u_prime) in self.V:
            if not np.any(np.all(b_prime == E, axis=1)):  # if b_prime not in the basis
                i = np.argmax([np.linalg.norm(b - e, ord=1) - np.linalg.norm(b_prime - e, ord=1) for e in E])
                w = np.array([np.linalg.norm(b - e, ord=1) for e in E])
                w[i] = np.linalg.norm(b - b_prime, ord=1)
                w /= np.sum(w)
                w = 1 - w
                alpha = np.array([V[e] for e in E])
                alpha[i] = u_prime
                u = min(u, np.dot(w, alpha))
        return u

    def __call__(self, b: np.ndarray) -> Any:
        return self.greedy(b)[0]


def SawtoothIteration(OfflinePlanningMethod): # TODO - Rethink having it inherit from Offline Planning Method
    def __init__(self, V: dict[np.ndarray, float], B: list[np.ndarray], k_max: int):
        super().__init__(k_max)
        self.V = V  # initial mapping from beliefs to utilities
        self.B = B  # beliefs to compute values including those in V map
    
    def solve(self, P: POMDP) -> SawtoothPolicy:
        E = np.eye(n)
        policy = SawtoothPolicy(P, self.V)
        for _ in range(self.k_max):
            V = {b: (self.V[b] if np.any(np.all(b == E, axis=1)) else policy.greedy(b)[1]) for b in self.B}
            policy = SawtoothPolicy(P, V)
        return policy


def random_belief_expansion(P: POMDP, B: list[np.ndarray]) -> list[np.ndarray]:
    B_prime = B.copy()
    for b in B:
        a = random.choices(P.A)
        b_prime, r = P.randstep(b, a)
        B_prime.append(b_prime)
    return list(set(B_prime))


def exploratory_belief_expansion(P: POMDP, B: list[np.ndarray]) -> list[np.ndarray]:
    B_prime = B.copy()
    for b in B:
        best = (b.copy(), 0.0)
        for a in P.A:
            b_prime, r = P.randstep(b, a)
            d = np.min([np.linalg.norm(b - b_prime, ord=1) for b in B_prime])
            if d > best[1]:
                best = (b_prime, d)
        B_prime.append(best[0])
    return list(set(B_prime))


class SawtoothHeuristicSearch(OfflinePlanningMethod): # TODO - Rethink having it inherit from Offline Planning Method
    def __init__(self, b: np.ndarray, delta: float, d: int, k_max: int, k_fib: int):
        self.b = B          # initial belief
        self.delta = delta  # gap threshold
        self.d = d          # depth
        self.k_max = k_max  # maximum number of iterations
        self.k_fib = k_fib  # number of iterations for fast informed bound

    def explore(self, P: POMDP, policy_hi: SawtoothPolicy, policy_lo: LookaheadAlphaVectorPolicy, b: np.ndarray, d: int = 0):
        pass  # TODO

    def solve(self, P: POMDP) -> LookaheadAlphaVectorPolicy:
        pass  # TODO


class TriangulatedPolicy(POMDPPolicy):
    pass  # TODO

class TriangulatedIteration(OfflinePlanningMethod): # TODO - Rethink inheritance
    pass

import cvxpy as cp
import itertools
import numpy as np

from abc import ABC, abstractmethod
from typing import Any

from ch19 import POMDP


class ConditionalPlan():
    def __init__(self, a: Any, subplans: dict[Any, 'ConditionalPlan'] = dict()):
        self.a = a                # action to take at root
        self.subplans = subplans  # dictionary mapping observations to subplans

    def __call__(self, o: Any = None) -> Any | 'ConditionalPlan':
        if o is None:
            return self.a
        return self.subplans[o]

    @staticmethod
    def create_for(P: POMDP, a: Any, plans: np.ndarray):
        # Note: Each row in plans is a ConditionalPlan
        subplans = {o: plan for (o, plan) in zip(P.O_space, plans)}
        return ConditionalPlan(a, subplans)


def evaluate_plan(P: POMDP, policy: ConditionalPlan, s: Any) -> float:
    def U(o, s_prime): return evaluate_plan(P, policy(o), s_prime)
    return P.R(s, policy()) if len(policy.subplans) == 0 else P.lookahead_from_state(U, s, policy())


def alphavector(P: POMDP, policy: ConditionalPlan) -> np.ndarray:
    return np.array([evaluate_plan(P, policy, s) for s in P.S])


class POMDPPolicy(ABC):
    def __init__(self, P: POMDP):
        self.P = P  # POMDP problem

    @abstractmethod
    def utility(self, b: np.ndarray) -> float:
        pass

    @abstractmethod
    def __call__(self, b: np.ndarray) -> Any:
        pass

    def greedy(self, b: np.ndarray) -> Any:
        def U(b): return self.utility(b)
        return self.P.greedy(U, b)


class AlphaVectorPolicy(POMDPPolicy):
    def __init__(self, P: POMDP, V: np.ndarray, a: list[Any]):
        super().__init__(P)
        self.V = V  # alpha vectors (each row is an alpha vector)
        self.a = a  # actions associated with alpha vectors

    def utility(self, b: np.ndarray) -> float:
        return np.max(self.V @ b)

    def __call__(self, b: np.ndarray) -> Any:
        i = np.argmax(self.V @ b)
        return self.a[i]


class LookaheadAlphaVectorPolicy(AlphaVectorPolicy):
    def __init__(self, P: POMDP, V: np.ndarray):
        super().__init__(P, V, a=None)

    def __call__(self, b: np.ndarray) -> Any:
        return self.greedy(b)[0]


def find_maximal_belief(alpha: np.ndarray, V: np.ndarray) -> np.ndarray:
    # Note: Each row in V is an alpha vector
    m = len(alpha)
    if len(V) == 0:
        return np.full(shape=m, fill_value=(1/m))  # arbitrary belief
    delta = cp.Variable()
    b = cp.Variable((m, 1))
    objective = cp.Maximize(delta)
    constraints = [b >= 0, cp.sum(b) == 1, b.T @ alpha >= V @ b + delta]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return b.value if delta.value > 0 else None


def find_dominating(V: np.ndarray) -> list[bool]:
    # Note: Each row in V is an alpha vector
    n = len(V)
    candidates, dominating = np.array([True] * n), np.array([False] * n)
    while np.any(candidates):
        i = np.argmax(candidates)
        b = find_maximal_belief(V[i], V[dominating])
        if b is None:
            candidates[i] = False
        else:
            k = np.argmax([np.dot(b, V[j]) if candidates[j] else -np.inf for j in range(n)])
            candidates[k], dominating[k] = False, True
    return dominating


def prune(plans: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Note: Each row in plans is a ConditionalPlan
    d = find_dominating(V)
    return (plans[d], V[d])


class POMDPSolutionMethod(ABC):
    pass


class ExactPlanningMethod(POMDPSolutionMethod):
    pass


class ValueIteration(ExactPlanningMethod):
    def __init__(self, k_max: int):  # TODO - Maybe, it should inherit from Offline planning method? Or k_max should be in POMDP Solution Method?
        self.k_max = k_max

    def solve(self, P: POMDP) -> LookaheadAlphaVectorPolicy:
        S, A, R = P.S, P.A, P.R
        plans = np.array([ConditionalPlan(a) for a in A])
        V = np.array([[R(s, a) for s in S] for a in A])
        plans, V = prune(plans, V)
        for _ in range(self.k_max - 1):
            plans, V = expand(plans, V, P)
            plans, V = prune(plans, V)
        return LookaheadAlphaVectorPolicy(P, V)


def expand(plans: np.ndarray, V: np.ndarray, P: POMDP) -> tuple[np.ndarray, np.ndarray]:
    plans_prime, V_prime = np.array([]), np.array([])
    for a in P.A:
        # Iterate over all possible mappings from observations to plans
        for inds in itertools.product(*[[i for i in range(len(plans))] for o in P.O_space]):
            plans_o = plans[[inds]]
            Vo = V[[inds]]
            policy = ConditionalPlan.create_for(P, a, plans_o)
            alpha = P.combine_alphavector(a, Vo)
            plans_prime.append(policy)
            V_prime = np.vstack([V_prime, [alpha]])
    return plans_prime, V_prime

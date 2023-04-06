import numpy as np

from abc import abstractmethod
from typing import Any, Callable

from ch07 import MDP, SolutionMethod


class OnlinePlanningMethod(SolutionMethod):
    @abstractmethod
    def __call__(self, s: Any) -> Any:
        pass


def rollout(P: MDP, s: Any, policy: Callable[[Any], Any], d: int) -> float:
    ret = 0.0
    for t in range(d):
        a = policy(s)
        s, r = P.randstep(s, a)
        ret += P.gamma**(t - 1) * r
    return ret


class RolloutLookahead(OnlinePlanningMethod):
    def __init__(self, P: MDP, policy: Callable[[Any], Any], d: int):
        self.P = P            # problem
        self.policy = policy  # rollout policy
        self.d = d            # depth

    def __call__(self, s: Any) -> Any:
        def U(s: Any): return rollout(self.P, s, self.policy, self.d)
        return (self.P.greedy(U, s))[0]


def forward_search(P: MDP, s: Any, d: int, U: Callable[[Any], float]) -> tuple[Any, float]:
    if d <= 0:
        return (None, U(s))
    best = (None, -np.inf)  # TODO - Decide if split up or not
    def U_prime(s: Any): return (forward_search(P, s, d - 1, U))[1]
    for a in P.A:
        u = P.lookahead(U_prime, s, a)
        if u > best[1]:
            best = (a, u)
    return best


class ForwardSearch(OnlinePlanningMethod):
    def __init__(self, P: MDP, d: int, U: Callable[[Any], float]):
        self.P = P  # problem
        self.d = d  # depth
        self.U = U  # value function at depth d

    def __call__(self, s: Any) -> Any:
        return (forward_search(self.P, s, self.d, self.U))[0]


def branch_and_bound(P: MDP, s: Any, d: int, 
                     U_lo: Callable[[Any], float], 
                     Q_hi: Callable[[Any, Any], float]) -> tuple[Any, float]:
    if d <= 0:
        return (None, U_lo(s))
    def U_prime(s: Any): return branch_and_bound(P, s, d - 1, U_lo, Q_hi)
    best = (None, -np.inf)  # TODO - Decide if split up or not
    for a in sorted(P.A, key=(lambda a: Q_hi(s, a)), reverse=True):
        if Q_hi(s, a) < best[1]:
            return best  # safe to prune
        u = P.lookahead(U_prime, s, a)
        if u > best[1]:
            best = (a, u)
    return best


class BranchAndBound(OnlinePlanningMethod):
    def __init__(self, P: MDP, d: int, U_lo: Callable[[Any], float], Q_hi: Callable[[Any, Any], float]):
        self.P = P        # problem
        self.d = d        # depth
        self.U_lo = U_lo  # lower bound on value function at depth d
        self.Q_hi = Q_hi  # upper bound on action value function

    def __call__(self, s: Any) -> Any:
        return (branch_and_bound(self.P, s, self.d, self.U_lo, self.Q_hi))[0]


def sparse_sampling(P: MDP, s: Any, d: int, m: int, U: Callable[[Any], float]):
    if d <= 0:
        return (None, U(s))
    best = (None, -np.inf)
    for a in P.A:
        u = 0.0
        for _ in range(m):
            s_prime, r = P.randstep(s, a)
            a_prime, u_prime = sparse_sampling(P, s_prime, d - 1, m, U)
            u += (r + P.gamma * u_prime) / m
        if u > best[1]:
            best = (a, u)
    return best


class SparseSampling(OnlinePlanningMethod):
    def __init__(self, P: MDP, s: Any, d: int, m: int, U: Callable[[Any], float]):
        self.P = P  # problem
        self.d = d  # depth
        self.m = m  # number of samples
        self.U = U  # value function at depth d

    def __call__(self, s: Any) -> Any:
        return (sparse_sampling(self.P, s, self.d, self.m, self.U))[0]


class MonteCarloTreeSearch(OnlinePlanningMethod):
    def __init__(self,
                 P: MDP,
                 N: dict[tuple[Any, Any], int],
                 Q: dict[tuple[Any, Any], float],
                 d: int,
                 m: int,
                 c: float,
                 U: Callable[[Any], float]):
        self.P = P  # problem
        self.N = N  # visit counts
        self.Q = Q  # action value estimates
        self.d = d  # depth
        self.m = m  # number of simulations
        self.c = c  # exploration constant
        self.U = U  # value function estimate

    def __call__(self, s: Any) -> Any:
        for _ in range(self.m):
            self.simulate(s, d=self.d)
        return self.P.A[np.argmax([self.Q[(s, a)] for a in self.P.A])]

    def simulate(self, s: Any, d: int):
        if d <= 0:
            return self.U(s)
        if (s, self.P.A[0]) not in self.N:
            for a in self.P.A:
                self.N[(s, a)] = 0
                self.Q[(s, a)] = 0.0
            return self.U(s)
        a = self.explore(s)
        s_prime, r = self.P.randstep(s, a)
        q = r + self.P.gamma * self.simulate(s_prime, d - 1)
        self.N[(s, a)] += 1
        self.Q[(s, a)] += (q - self.Q[(s, a)]) / self.N[(s, a)]
        return q

    def explore(self, s: Any) -> Any:
        A, N, Q, c = self.P.A, self.N, self.Q, self.c
        Ns = np.sum([N[(s, a)] for a in A])
        return A[np.argmax([Q[(s, a)] + c*self.bonus(N[(s, a)], Ns) for a in A])]

    @staticmethod
    def bonus(Nsa: int, Ns: int) -> float:
        return np.inf if Nsa == 0 else np.sqrt(np.log(Ns)/Nsa)


class HeuristicSearch(OnlinePlanningMethod):
    def __init__(self, P: MDP, U_hi: Callable[[Any], float], d: int, m: int):
        self.P = P        # problem
        self.U_hi = U_hi  # upper bound on value function
        self.d = d        # depth
        self.m = m        # number of simulations

    def __call__(self, s: int | np.ndarray) -> Any:
        U = np.array([self.U_hi(s) for s in self.P.S])
        for _ in range(self.m):
            self.simulate(U, s)
        return self.P.greedy(U, s)[0]

    def simulate(self, U: np.ndarray, s: int | np.ndarray):
        for _ in range(self.d):
            a, u = self.P.greedy(U, s)
            U[s] = u 
            s = np.random.choice(self.P.S, p=[self.P.T(s, a, s_prime) for s_prime in self.P.S])


class LabeledHeuristicSearch(OnlinePlanningMethod):
    def __init__(self, P: MDP, U_hi: Callable[[Any], float], d: int, delta: float):
        self.P = P          # problem 
        self.U_hi = U_hi    # upper bound on value function
        self.d = d          # depth
        self.delta = delta  # gap threshold

    def __call__(self, s: int | np.ndarray) -> Any:
        U, solved = np.array([self.U_hi(s) for s in self.P.S]), set() 
        while s not in solved:
            self.simulate(U, solved, s)
        return self.P.greedy(U, s)[0]

    def simulate(self, U: np.ndarray, solved: set[int] | set[np.ndarray], s: int | np.ndarray):
        visited = []
        for _ in range(self.d):
            if s in solved:
                break
            visited.append(s)
            a, u = self.P.greedy(U, s)
            U[s] = u 
            s = np.random.choice(self.P.S, p=[self.P.T(s, a, s_prime) for s_prime in self.P.S])
        while len(visited) != 0:
            if self.label(U, solved, visited.pop()):
                break
        
    def label(self, U: np.ndarray, solved: set[int] | set[np.ndarray], s: int | np.ndarray):
        if s in solved:
            return False
        found, envelope = self.expand(U, solved, s)
        if found:
            for s in reversed(envelope):
                U[s] = self.P.greedy(U, s)[1]
        else:
            solved = solved.union(envelope)
        return found

    def expand(self, U: np.ndarray, solved: set[int] | set[np.ndarray],
               s: int | np.ndarray) -> tuple[bool, list[int] | list[np.ndarray]]:
        found, to_expand, envelope = False, {s}, []
        while len(to_expand) != 0:
            s = to_expand.pop() # TODO - set.pop() removes a random element, whereas it is unknown what Julia's pop!(set) removes
            envelope.append(s)
            a, u  = self.P.greedy(U, s)
            if np.abs(U[s] - u) > self.delta:
                found = True
            else:
                for s_prime in self.P.S:
                    if (self.P.T(s, a, s_prime) > 0) and (s_prime not in solved.union(envelope)):
                        to_expand.add(s_prime)
        return found, envelope

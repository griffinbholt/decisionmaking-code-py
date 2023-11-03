import numpy as np

from abc import ABC, abstractmethod
from queue import PriorityQueue
from scipy.stats import dirichlet
from typing import Callable

from ch07 import MDP, LinearProgramFormulation


class RLMDP(ABC):
    def __init__(self, A: list[int], gamma: float):
        self.A = A          # action space (assumes 1:nactions)
        self.gamma = gamma  # discount factor

    @abstractmethod
    def lookahead(self, s: int, a: int) -> float:
        pass

    @abstractmethod
    def update(self, s: int, a: int, r: float, s_prime: int):
        pass


class ModelBasedMDP(RLMDP):
    def __init__(self, S: list[int], A: list[int], gamma: float, U: np.ndarray, planner: 'MLEModelUpdate'):
        super().__init__(A, gamma)
        self.S = S  # state space (assumes 1:nstates)
        self.U = U  # value function
        self.planner = planner
    
    @abstractmethod
    def update(self, s: int, a: int, r: float, s_prime: int):
        pass

    @abstractmethod
    def to_MDP(self) -> MDP:
        pass

    def simulate(self, policy: 'RLPolicy', h: int, s: int):
        for _ in range(h):
            P = self.to_MDP()
            a = policy(self, s)
            s_prime, r = P.TR(s, a)
            self.update(s, a, r, s_prime)
            s = s_prime


class MaximumLikelihoodMDP(ModelBasedMDP):
    def __init__(self,
                 S: list[int],
                 A: list[int],
                 N: np.ndarray,
                 rho: np.ndarray,
                 gamma: float,
                 U: np.ndarray,
                 planner: 'MLEModelUpdate'):
        super().__init__(S, A, gamma, U, planner)
        self.N = N      # transition count N(s, a, s')
        self.rho = rho  # reward sum p(s, a)

    def lookahead(self, s: int, a: int) -> float:
        n = self.N[s, a].sum()
        if n == 0:
            return 0.0
        r = self.rho[s, a] / n
        def T(s, a, s_prime): return self.N[s, a, s_prime] / n
        return r + self.gamma * np.sum([T(s, a, s_prime)*self.U[s_prime] for s_prime in self.S])

    def backup(self, U: np.ndarray, s: int) -> float:
        return np.max([self.lookahead(s, a) for a in self.A])

    def update(self, s: int, a: int, r: float, s_prime: int):
        self.N[s, a, s_prime] += 1
        self.rho[s, a] += r
        self.planner.update(self, s, a, r, s_prime)

    def to_MDP(self) -> MDP:
        N_sa = np.sum(self.N, axis=2, keepdims=True)
        T = np.divide(self.N, N_sa, out=np.zeros_like(self.N), where=(N_sa != 0))
        N_sa = np.squeeze(N_sa)
        R = np.divide(self.rho, N_sa, out=np.zeros_like(self.rho), where=(N_sa != 0))
        return MDP(self.gamma, self.S, self.A, T, R)


class RLPolicy(ABC):
    @abstractmethod
    def __call__(self, model: ModelBasedMDP, s: int) -> int:
        pass

class EpsilonGreedyExploration(RLPolicy):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon  # probability of random arm

    def __call__(self, model: ModelBasedMDP, s: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.choice(model.A)
        def Q(s, a): return model.lookahead(s, a)
        return np.argmax([Q(s, a) for a in model.A])  


class ModelUpdateScheme(ABC):
    @abstractmethod
    def update(self, model: ModelBasedMDP, s: int, a: int, r: float, s_prime: int):
        pass


class MLEModelUpdate(ModelUpdateScheme):
    @abstractmethod
    def update(self, model: MaximumLikelihoodMDP, s: int, a: int, r: float, s_prime: int):
        pass


class FullUpdate(MLEModelUpdate):
    def __init__(self):
        self.solver = LinearProgramFormulation()

    def update(self, model: MaximumLikelihoodMDP, s: int, a: int, r: float, s_prime: int):
        P = model.to_MDP()
        U = self.solver.solve(P)
        np.copyto(model.U, U)


class RandomizedUpdate(MLEModelUpdate):
    def __init__(self, m: int):
        self.m = m

    def update(self, model: MaximumLikelihoodMDP, s: int, a: int, r: float, s_prime: int):
        U = model.U
        U[s] = model.backup(U, s)
        for _ in range(self.m):
            s = np.random.randint(low=0, high=len(model.S))
            U[s] = model.backup(U, s)


class PrioritizedUpdate(MLEModelUpdate):
    def __init__(self, m: int, pq: PriorityQueue):
        self.m = m    # number of updates
        self.pq = pq  # priority queue

    def update(self, model: ModelBasedMDP, s: int, a: int, r: float, s_prime: int):
        self.pq.put((np.inf, s))
        for _ in range(self.m):
            if self.pq.empty():
                break
            self.update_state(model, self.pq.get()[1])

    def update_state(self, model: ModelBasedMDP, s: int):
        u = model.U[s]
        model.U[s] = model.backup(model.U, s)
        for s_other in model.S:
            for a_other in model.A:
                n_sa = model.N[s_other, a_other].sum()
                if n_sa > 0:
                    T = model.N[s_other, a_other, s] / n_sa
                    priority = T * np.abs(model.U[s] - u)
                    if priority > 0:
                        curr_priority = self._get_current_priority(s_other, default_val=0.0)
                        new_priority = max(curr_priority, priority)
                        self.pq.put((new_priority, s_other))

    def _get_current_priority(self, s: int, default_val: float) -> float:
        for i in range(self.pq.qsize()):
            priority, item = self.pq.queue[i]
            if item == s:
                del self.pq.queue[i]
                return priority
        return default_val


class RmaxMDP(MaximumLikelihoodMDP):
    def __init__(self,
                 S: list[int],
                 A: list[int],
                 N: np.ndarray,
                 rho: np.ndarray,
                 gamma: float,
                 U: np.ndarray,
                 planner: MLEModelUpdate,
                 m: int,
                 rmax: float):
        super().__init__(S, A, N, rho, gamma, U, planner)
        self.m = m        # count threshold
        self.rmax = rmax  # maximum reward

    def lookahead(self, s: int, a: int) -> float:
        n = self.N[s, a].sum()
        if n < self.m:
            return self.rmax / (1 - self.gamma)
        r = self.rho[s, a] / n
        def T(s, a, s_prime): return self.N[s, a, s_prime] / n
        return r + self.gamma * np.sum([T(s, a, s_prime)*self.U[s_prime] for s_prime in self.S])

    def to_MDP(self) -> MDP:
        N_sa = np.sum(self.N, axis=2, keepdims=True)
        T = np.divide(self.N, N_sa, out=np.zeros_like(self.N), where=(N_sa >= self.m))
        N_sa = np.squeeze(N_sa)
        R = np.divide(self.rho, N_sa, out=np.full_like(self.rho, self.rmax), where=(N_sa >= self.m))
        for s in np.where(N_sa < self.m)[0]:
            T[s, :, s] = 1.0
        return MDP(self.gamma, self.S, self.A, T, R)


class BayesianMDP(ModelBasedMDP):
    def __init__(self,
                 S: list[int],
                 A: list[int],
                 D: np.ndarray,
                 R: Callable[[int, int], float],
                 gamma: float, U: np.ndarray,
                 planner: MLEModelUpdate):
        super().__init__(S, A, gamma, U, planner)
        self.D = D          # Dirichlet distributions D[s, a]
        self.R = R          # reward function as matrix (not estimated)
        self.gamma = gamma  # discount
        self.U = U          # value function
        self.planner = planner

    def lookahead(self, s: int, a: int) -> float:
        n = np.sum(self.D[s, a].alpha)
        if n == 0:
            return 0.0
        r = self.R(s, a)
        def T(s, a, s_prime): return self.D[s, a].alpha[s_prime] / n
        return r + self.gamma * np.sum([T(s, a, s_prime)*self.U[s_prime] for s_prime in self.S])

    def update(self, s: int, a: int, r: float, s_prime: int):
        alpha = self.D[s, a].alpha
        alpha[s_prime] += 1
        self.D[s, a] = dirichlet(alpha)
        self.planner.update(self, s, a, r, s_prime)

    def to_MDP(self) -> MDP:  # Accomplished via sampling
        T = np.array([[self.D[s, a].rvs()[0] for a in self.A] for s in self.S])
        return MDP(self.gamma, self.S, self.A, T, self.R)


class PosteriorSamplingUpdate(ModelUpdateScheme):
    def __init__(self):
        self.solver = LinearProgramFormulation()

    def update(self, model: ModelBasedMDP, s: int, a: int, r: float, s_prime: int):
        P = model.sample()
        U = self.solver.solve(P)
        np.copyto(model.U, U)

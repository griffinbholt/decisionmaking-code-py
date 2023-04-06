import numpy as np 

from abc import ABC, abstractmethod
from scipy.stats import beta

from convenience import normalize

class BanditProblem():
    def __init__(self, theta: np.ndarray):
        self.theta = theta

    def R(self, a: int):
        return 1 if np.random.rand() < self.theta[a] else 0

    def simulate(self, model: BanditModel, policy: BanditPolicy, h: int):
        for _ in range(h):
            a = policy(model)
            r = self.R(a)
            model.update(a, r)

class BanditModel():
    def __init__(self, B: list[beta]):
        self.B = B

    def update(self, a: int, r: int) -> beta:
        alpha, b = model.B[a].args
        model.B[a] = beta(alpha + r, b + (1 - r))
        return model

class BanditPolicy(ABC):
    @abstractmethod
    def __call__(self, model: BanditModel) -> int:
        pass

class EpsilonGreedyExploration(BanditPolicy):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon # probability of random arm

    def __call__(self, model: BanditModel) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(low=0, high=len(model.B))
        return np.argmax([distrib.mean() for distrib in model.B])

class ExploreThenCommitExploration(BanditPolicy):
    def __init__(self, k: int):
        self.k = k # pulls remaining until commitment

    def __call__(self, model: BanditModel) -> int:
        if self.k > 0
            self.k -= 1
            return np.random.randint(low=0, high=len(model.B))
        return np.argmax([distrib.mean() for distrib in model.B])

class SoftmaxExploration(BanditPolicy):
    def __init__(self, lam: float, alpha: float):
        self.lam = lam # precision parameter
        self.alpha = alpha # precision factor

    def __call__(self, model: BanditModel) -> int:
        p = np.array([distrib.mean() for distrib in model.B])
        weights = normalize(np.exp(self.lam * p), ord=1)
        self.lam *= self.alpha
        return np.random.choice(len(model.B), p=weights)

class QuantileExploration(BanditPolicy):
    def __init__(self, alpha: float):
        self.alpha = alpha # quantile (e.g., 0.95)

    def __call__(self, model: BanditModel) -> int:
        return np.argmax([distrib.ppf(self.alpha) for distrib in model.B])

class UCB1Exploration(BanditPolicy):
    def __init__(self, c: float):
        self.c = c # exploration constant

    def __call__(self, model: BanditModel) -> int:
        p = np.array([distrib.mean() for distrib in model.B])
        u = p + np.array([self.bonus(model.B, a) for a in range(len(model.B))])
        return np.argmax(u)

    def bonus(self, B: list[beta], a: int):
        N = np.sum([np.sum(distrib.args) for distrib in B])
        Na = np.sum(B[a].args)
        return self.c * np.sqrt(np.log(N)/Na)

class PosteriorSamplingExploration(BanditPolicy):
    def __call__(self, model: BanditModel) -> int:
        return np.argmax([distrib.rvs() for distrib in model.B])

# TODO - Decide how to incorporate the simulate function
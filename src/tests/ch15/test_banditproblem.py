import sys; sys.path.append('./code/'); sys.path.append('../../')

import numpy as np
import pytest
from scipy.stats import beta

from ch15 import BanditProblem, BanditModel, BanditPolicy, EpsilonGreedyExploration, ExploreThenCommitExploration, SoftmaxExploration, QuantileExploration, UCB1Exploration, PosteriorSamplingExploration


class TestBanditProblem():
    n_arms = 5
    problem = BanditProblem(theta=np.array([0.25, 0.3, 0.05, 0.75, 0.25]))
    model = None
    tol = 0.5

    @pytest.fixture(autouse=True)
    def run_before(self):  # Reset the prior to uniform distribution
        self.model = BanditModel(B=[beta(1, 1) for _ in range(self.n_arms)])
        yield

    def test_reward(self):
        for i in range(self.n_arms):
            rewards = [self.problem.R(i) for _ in range(10000)]
            assert(np.abs(np.mean(rewards) - self.problem.theta[i]) < self.tol)

    def test_epsilon_greedy(self):
        eps_greedy = EpsilonGreedyExploration(epsilon=0.3)
        self.run_policy(eps_greedy, horizon=10000)
        freqs = np.bincount([eps_greedy(self.model) for _ in range(1000)]) / 1000.0
        assert(freqs.argmax() == 3)
        assert(np.abs(freqs[3] - 0.7) < 0.1)

    def test_explore_then_commit(self):
        explore_then_commit = ExploreThenCommitExploration(k=1000)
        self.run_policy(explore_then_commit, horizon=10000)
        assert(explore_then_commit(self.model) == 3)
    
    def test_softmax(self):
        softmax = SoftmaxExploration(lam=0.001, alpha=1.01)
        self.run_policy(softmax, horizon=300)
        freqs = np.bincount([softmax(self.model) for _ in range(1000)]) / 1000.0
        assert(freqs.argmax() == 3)

    def test_quantile(self):
        quantile_expl = QuantileExploration(alpha=0.5)
        self.run_policy(quantile_expl, horizon=10)
    
    def test_ucb1(self):
        ucb1_expl = UCB1Exploration(c=5)
        self.run_policy(ucb1_expl, horizon=100)

    def test_posterior_sampling(self):
        posterior_sampling = PosteriorSamplingExploration()
        self.run_policy(posterior_sampling, horizon=1000)

    def run_policy(self, policy: BanditPolicy, horizon: int):
        self.problem.simulate(model=self.model, policy=policy, h=horizon)
        for i in range(self.n_arms):
            assert(np.abs(self.model.B[i].mean() - self.problem.theta[i]) < self.tol)

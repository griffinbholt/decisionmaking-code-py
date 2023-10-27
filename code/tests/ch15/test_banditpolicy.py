import sys; sys.path.append('./code/'); sys.path.append('../../')

import numpy as np

from scipy.stats import beta

from ch15 import BanditModel, EpsilonGreedyExploration, ExploreThenCommitExploration, SoftmaxExploration, QuantileExploration, UCB1Exploration, PosteriorSamplingExploration


class TestBanditPolicy():
    def test_epsilon_greedy(self):
        n_arms = 2
        model = BanditModel(B=[beta(1, 1) for _ in range(n_arms)])
        policy = EpsilonGreedyExploration(epsilon=0.3)
        
        action = policy(model)
        print(action)

    def test_explore_then_commit(self):
        pass  # TODO

    def test_softmax(self):
        pass  # TODO

    def test_quantile(self):
        pass  # TODO

    def test_ucb1(self):
        pass  # TODO

    def test_posterior_sampling(self):
        pass  # TODO

test = TestBanditPolicy()
test.test_epsilon_greedy()
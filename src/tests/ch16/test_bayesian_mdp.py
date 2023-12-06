import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np
import random

from scipy.stats import dirichlet

from ch07 import MDP
from ch16 import BayesianMDP, EpsilonGreedyExploration, PosteriorSamplingUpdate
from problems.HexWorldMDP import HexWorld

class TestBayesianMDP():
    def test_posterior_sampling_update(self):
        P = HexWorld
        D = np.array([[dirichlet(np.ones(len(P.S))) for _ in P.A] for _ in P.S])
        U = np.zeros(len(P.S))
        planner = PosteriorSamplingUpdate()
        model = BayesianMDP(P.S, P.A, D, P.R, P.gamma, U, planner)
        policy = EpsilonGreedyExploration(epsilon=0.1)
        try:
            model.simulate(policy, h=100, s=random.choices(P.S)[0])
        except Exception as exc:
            assert False, "{exc}"

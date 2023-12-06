import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np
import random

from queue import PriorityQueue
from typing import Any

from ch07 import MDP
from ch16 import MaximumLikelihoodMDP, RLPolicy, EpsilonGreedyExploration, FullUpdate, RandomizedUpdate, PrioritizedUpdate
from problems.HexWorldMDP import HexWorld
from simulate_returns import simulate_returns


class TestMaximumLikelihoodMDP():
    def test_full_update(self):
        P = HexWorld
        N = np.zeros((len(P.S), len(P.A), len(P.S)))
        rho = np.zeros((len(P.S), len(P.A)))
        U = np.zeros(len(P.S))
        planner = FullUpdate()
        model = MaximumLikelihoodMDP(P.S, P.A, N, rho, P.gamma, U, planner)
        policy = EpsilonGreedyExploration(epsilon=0.1)
        try:
            model.simulate(policy, h=100, s=random.choices(P.S)[0])
        except Exception as exc:
            assert False, "{exc}"
    
    def test_random_update(self):
        random.seed(0); np.random.seed(0)
        P = HexWorld
        N = np.zeros((len(P.S), len(P.A), len(P.S)))
        rho = np.zeros((len(P.S), len(P.A)))
        U = np.zeros(len(P.S))
        planner = RandomizedUpdate(m=10)
        model = MaximumLikelihoodMDP(P.S, P.A, N, rho, P.gamma, U, planner)
        policy = EpsilonGreedyExploration(epsilon=0.1)
        assert(np.isclose(np.mean(simulate_returns(P, model, policy, h=10, n=1000)), 4.5562, atol=0.1))

    def test_prioritized_update(self):
        P = HexWorld
        N = np.zeros((len(P.S), len(P.A), len(P.S)))
        rho = np.zeros((len(P.S), len(P.A)))
        U = np.zeros(len(P.S))
        planner = PrioritizedUpdate(m=5, pq=PriorityQueue())
        model = MaximumLikelihoodMDP(P.S, P.A, N, rho, P.gamma, U, planner)
        policy = EpsilonGreedyExploration(epsilon=0.1)
        try:
            model.simulate(policy, h=100, s=random.choices(P.S)[0])
        except Exception as exc:
            assert False, "{exc}"
import sys; sys.path.append('./src/'); sys.path.append('../'); sys.path.append('../../')

import numpy as np
import random

from ch07 import MDP
from ch09 import rollout, MonteCarloTreeSearch
from problems.HexWorldMDP import HexWorld

class TestMonteCarloTreeSearch():
    def test_explore(self):
        A = [0, 1]
        P = MDP(None, None, A, None, None, None)
        c = 2.5
        d = 5 # doesn't actually matter for this test
        m = 1 # doesn't matter
        policy = MonteCarloTreeSearch(P, {}, {}, d, m, c, lambda s: rollout(P, s, lambda x: 1, d))
        
        # With zero visits, we return the first action
        policy.N[(0, 0)] = 0
        policy.N[(0, 1)] = 0
        policy.Q[(0, 0)] = 0.0
        policy.Q[(0, 1)] = 0.0
        assert(policy.explore(0) == 0)

        # With zero visits to action 2, we return the 2nd action
        policy.N[(0, 0)] = 1
        policy.N[(0, 1)] = 0
        policy.Q[(0, 0)] = 999.0 # Despite large value
        policy.Q[(0, 1)] = 0.0
        assert(policy.explore(0) == 1)

        # Test that exploration constant matters
        policy.N[(0, 0)] = 1
        policy.N[(0, 1)] = 3
        policy.Q[(0, 0)] = 10.0 # value is 10 + c*sqrt(log(4)/1) ~ 12.9
        policy.Q[(0, 1)] = 11.3 # value is 11.3 + c*sqrt(log(4)/3) ~ 13.0
        assert(policy.explore(0) == 1)
        policy.Q[(0, 1)] = 11.0 # value is 11.0 + c*sqrt(log(4)/3) ~ 12.7
        assert(policy.explore(0) == 0)

    def test_simulate(self):
        d = 2
        c = 1.0
        r = 0.5
        m = 10
        P = MDP(0.9, [0, 1], [0, 1], None, None, lambda s, a: (np.random.randint(2), r))

        mcts = MonteCarloTreeSearch(
            P,
            {},
            {},
            d,
            m,
            c,
            lambda s: rollout(P, s, lambda x: np.random.randint(2), d)
        )

        # Expand the first node
        q = mcts.simulate(s=0, d=10)
        assert(len(mcts.N) == 2)
        assert(len(mcts.Q) == 2)
        assert(mcts.N[(0, 0)] == 0)
        assert(mcts.N[(0, 1)] == 0)
        assert(mcts.Q[(0, 0)] == 0.0)
        assert(mcts.Q[(0, 1)] == 0.0)
    
    def test_full(self):
        random.seed(42); np.random.seed(42)
        d = 2
        c = 1.0
        m = 20
        P = HexWorld

        mcts = MonteCarloTreeSearch(
            P,
            {},
            {},
            d,
            m,
            c,
            lambda s: rollout(P, s, P.random_policy(), d)
        )

        # Smoke test
        actions = [mcts(s) for s in P.S]

        # Regression test
        assert(actions[1] == 2)
        assert(actions[2] == 0)
        assert(actions[-1] == 0)
        assert(actions[-2] == 0)
        assert(actions[-3] == 5)
        assert(actions[-4] == 0)
        assert(actions[-5] == 1)

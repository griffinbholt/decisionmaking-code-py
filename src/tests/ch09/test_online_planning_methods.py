import sys; sys.path.append('./src/'); sys.path.append('../'); sys.path.append('../../')

import numpy as np
import random

from ch09 import rollout, RolloutLookahead, forward_search, ForwardSearch, branch_and_bound, BranchAndBound, sparse_sampling, SparseSampling
from problems.HexWorldMDP import HexWorld, StraightLineHexWorld

class TestOnlinePlanningMethods():
    def test_rollout(self):
        P = HexWorld
        assert(np.abs(-2.025 - np.mean([rollout(P, s, lambda s: random.choices(P.A)[0], d=10) for s in P.S for _ in range(10000)])) < 1e-2)

    def test_forward_search(self):
        P = StraightLineHexWorld
        def U(s): return 0.0

        # Search from leftmost state to depth 1 yields no reward, take action 1
        best_a, best_u = forward_search(P, 0, 1, U)
        assert(best_a == 0)
        assert(np.abs(best_u - (-0.3)) < 1e-10)

        # Search from state 3 to depth 2 yields no reward, take action 1
        best_a, best_u = forward_search(P, 2, 2, U)
        assert(best_a == 0)
        assert(np.abs(best_u - (-0.57)) < 1e-10)

        # Search from state 3 to depth 1 with reward to right should take action 4
        best_a, best_u = forward_search(P, 2, 1, lambda s: 5.0 if s == 1 else 0.0)
        assert(best_a == 3)
        assert(np.abs(best_u - (2.85)) < 1e-10)

        # Search from state 3 to depth 2 with reward to right should take action 4
        best_a, best_u = forward_search(P, 2, 2, lambda s: 5.0 if s == 1 else 0.0)
        assert(best_a == 3)
        assert(np.abs(best_u - (2.34375)) < 1e-10)  # NOTE: We can get reward via multiple paths

        # Search from state 5 to depth 3 should yield reward, with action 1
        best_a, best_u = forward_search(P, 4, 3, U)
        assert(best_a == 0)
        assert(np.abs(best_u - (3.27507)) < 1e-10)
    
    def test_forward_search_planning_method(self):
        P = StraightLineHexWorld
        def U(s): return 0.0

        # Search from leftmost state to depth 1 yields no reward, take action 1
        policy = ForwardSearch(P, 1, U)
        assert(policy(0) == 0)

        # Search from state 3 to depth 2 yields no reward, take action 1
        policy = ForwardSearch(P, 2, U)
        assert(policy(2) == 0)

        # Search from state 3 to depth 1 with reward to right should take action 4
        policy = ForwardSearch(P, 1, lambda s: 5.0 if s == 1 else 0.0)
        assert(policy(2) == 3)

        # Search from state 3 to depth 2 with reward to right should take action 4
        policy = ForwardSearch(P, 2, lambda s: 5.0 if s == 1 else 0.0)
        assert(policy(2) == 3)

        # Search from state 5 to depth 3 should yield reward, with action 1
        policy = ForwardSearch(P, 3, U)
        assert(policy(4) == 0)

    def test_branch_and_bound(self):
        P = StraightLineHexWorld
        def Qhi(s, a): return 1000.0
        def Ulo(s): return 0.0

        # Search from leftmost state to depth 1 yields no reward, take action 1
        policy = BranchAndBound(P, 1, Ulo, Qhi)
        assert(policy(0) == 0)

        # Search from state 3 to depth 2 yields no reward, take action 1
        policy = BranchAndBound(P, 2, Ulo, Qhi)
        assert(policy(2) == 0)

        # Search from state 3 to depth 1 with reward to right should take action 4
        policy = BranchAndBound(P, 1, lambda s: 5.0 if s == 1 else 0.0, Qhi)
        assert(policy(2) == 3)

        # Search from state 3 to depth 2 with reward to right should take action 4
        policy = BranchAndBound(P, 2, lambda s: 5.0 if s == 1 else 0.0, Qhi)
        assert(policy(2) == 3)

        # Search from state 5 to depth 3 should yield reward, with action 1
        policy = BranchAndBound(P, 3, Ulo, Qhi)
        assert(policy(4) == 0)

    def test_sparse_sampling(self):
        P = StraightLineHexWorld
        def U(s): return 0.0

        np.random.seed(5)

        # Search from leftmost state to depth 1 yields no reward, take action 1
        best_a, best_u = sparse_sampling(P, 0, 1, 50, U)
        assert(best_a == 0)
        assert(np.abs(best_u - (-0.3)) < 1e-10)

        # Search from state 3 to depth 2 yields no reward, take action 1
        best_a, best_u = sparse_sampling(P, 2, 2, 50, U)
        assert(best_a == 0)
        assert(np.abs(best_u - (-0.57)) < 1e-10)

        # Search from state 3 to depth 1 with reward to right should take action 4
        best_a, best_u = sparse_sampling(P, 2, 1, 50, lambda s: 5.0 if s == 1 else 0.0)
        assert(best_a == 3)
        assert(np.abs(best_u - (2.94)) < 1e-10)

        # Search from state 3 to depth 2 with reward to right should take action 4
        best_a, best_u = sparse_sampling(P, 2, 2, 50, lambda s: 5.0 if s == 1 else 0.0)
        assert(best_a == 3)
        assert(np.abs(best_u - (2.52888)) < 5e-2)  # NOTE: We can get reward via multiple paths

        # Search from state 3 to depth 2 should yield reward, with action 1
        policy = SparseSampling(P, 2, 50, lambda s: 5.0 if s == 1 else 0.0)
        assert(policy(2) == 3)

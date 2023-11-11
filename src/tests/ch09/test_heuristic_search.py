import sys; sys.path.append('./src/'); sys.path.append('../'); sys.path.append('../../')

import numpy as np

from ch09 import HeuristicSearch, LabeledHeuristicSearch
from problems.HexWorldMDP import StraightLineHexWorld

class TestHeuristicSearch():
    def test_heuristic_search(self):
        P = StraightLineHexWorld
        def Uhi(s): return 11.0
        d = 10
        m = 100
        s0 = 0

        hs_policy = HeuristicSearch(P, Uhi, d, m)
        
        policy = hs_policy.extract_policy(s0)

        U_expected = [2.36476, 3.2223, 4.211, 5.356, 6.682, 8.219, 10.0, 0.0]
        assert(np.all(np.isclose(policy.U - U_expected, 0, atol=0.015)))

        for s in range(8):
            assert(policy(s) == 0)
    
    def test_expand_label(self):
        P = StraightLineHexWorld
        def Uhi(s): return 11.0
        delta = 1e-5
        d = 10
        s0 = 0

        hs_policy = LabeledHeuristicSearch(P, Uhi, d, delta)
        U, solved = np.array([Uhi(s) for s in P.S]), set()

        # First call only updates the first state.
        found, envelope = hs_policy.expand(U, solved, s0)
        assert(found)
        assert(envelope == [0])

        assert(hs_policy.label(U, solved, s0))
        assert(np.all(np.isclose(U - np.array([9.6, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0]), 0, atol=1e-10)))
        assert(len(solved) == 0)

        # Second call updates the first two states.
        found, envelope = hs_policy.expand(U, solved, s0)
        assert(found)
        assert(envelope == [0])

        assert(hs_policy.label(U, solved, s0))
        assert(np.all(np.isclose(U - np.array([9.222, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0]), 0, atol=1e-10)))
        assert(len(solved) == 0)

        # Further runs converge on the optimal value function
        while hs_policy.label(U, solved, s0):
            pass
        found, envelope = hs_policy.expand(U, solved, s0)
        assert(not found)
        assert(envelope == [0])
        assert(np.all(np.isclose(U - np.array([2.3754520253300715, 3.2284830477016877, 4.2168918894329535, 5.362164436992639, 6.689196738495563, 8.226836266888837, 10.0085090941116, 0.009454549012888593]), 0, atol=1e-2)))
        
        # A further call on any state returns labeled.
        assert(np.all([s in solved for s in P.S]))
        assert(np.all([not hs_policy.label(U, solved, s) for s in P.S]))

    def test_labeled_heuristic_search(self):
        P = StraightLineHexWorld
        def Uhi(s): return 11.0
        delta = 1e-5
        d = 10
        s0 = 0

        hs_policy = LabeledHeuristicSearch(P, Uhi, d, delta)
        
        policy = hs_policy.extract_policy(s0)

        U_expected = [2.36476, 3.2223, 4.211, 5.356, 6.682, 8.219, 10.0, 0.0]
        assert(np.all(np.isclose(policy.U - U_expected, 0, atol=0.01)))

        for s in range(8):
            assert(policy(s) == 0)

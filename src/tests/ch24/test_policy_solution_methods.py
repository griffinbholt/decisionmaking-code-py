import sys; sys.path.append('./src/'); sys.path.append('../'); sys.path.append('../../')

import numpy as np

from problems.SimpleGames import PrisonersDilemma, RockPaperScissors, TravelersDilemma
from ch24 import joint_action, NashEquilibrium, CorrelatedEquilibrium, IteratedBestResponse, HierarchicalSoftmax

class TestPolicySolutionMethods():
    def test_nash_equilibrium(self):
        P = PrisonersDilemma()
        M = NashEquilibrium()
        policy = M.solve(P)
        assert(np.isclose(policy[0]('cooperate'), 0.0))
        assert(np.isclose(policy[0]('defect'), 1.0))
        assert(np.isclose(policy[1]('cooperate'), 0.0))
        assert(np.isclose(policy[1]('defect'), 1.0))

    def test_correlated_equilibrium(self):
        P = RockPaperScissors()
        M = CorrelatedEquilibrium()
        policy = M.solve(P)
        for a in joint_action(P.A):
            assert(np.isclose(policy(a), 1/9))

    def test_iterated_best_response(self):
        np.random.seed(0)
        P = TravelersDilemma()
        M = IteratedBestResponse.create_from_game(P, 2)
        assert(np.isclose(M.solve(P)[0](45), 0))

    def test_hierarchical_softmax(self):
        np.random.seed(0)
        P = TravelersDilemma()
        M = HierarchicalSoftmax.create_from_game(P, 5, 10)
        assert(np.isclose(M.solve(P)[0](45), 0))

import sys; sys.path.append('./src/'); sys.path.append('../'); sys.path.append('../../')

import numpy as np

from problems.SimpleGames import PrisonersDilemma
from ch24 import SimpleGamePolicy
class TestSimpleGame():
    P = PrisonersDilemma()

    def test_utility(self):
        policy = [SimpleGamePolicy(a) for a in self.P.A[0]]
        assert(np.isclose(self.P.utility(policy, 0), -4.0))
        assert(np.isclose(self.P.utility(policy, 1), 0.0))

        policy_opt = [SimpleGamePolicy('cooperate') for _ in self.P.I]
        assert(np.isclose(self.P.utility(policy_opt, 0), -1.0))
        assert(np.isclose(self.P.utility(policy_opt, 1), -1.0))

    def test_best_response(self):
        policy = [SimpleGamePolicy('cooperate') for _ in self.P.I]
        policy_1 = self.P.best_response(policy, 0)
        assert(np.isclose(policy_1("cooperate"), 0.0))
        assert(np.isclose(policy_1("defect"), 1.0))

    def test_softmax_response(self):
        policy = [SimpleGamePolicy('cooperate') for _ in self.P.I]
        policy_2 = self.P.softmax_response(policy, i=1, lam=0.5)
        assert(np.isclose(policy_2("cooperate"), 0.37754066879814546))

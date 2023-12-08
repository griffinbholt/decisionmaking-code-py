import sys; sys.path.append('./src/'); sys.path.append('../'); sys.path.append('../../')

import numpy as np

from problems.SimpleGames import PrisonersDilemma, RockPaperScissors
from ch24 import FictitiousPlay, GradientAscent


class TestSimulativeMethods():
    def test_fictitious_play(self):
        P = PrisonersDilemma()
        policy = [FictitiousPlay.create_from_game(P, i) for i in P.I]
        try:
            policy = P.simulate(policy, 3)
        except Exception as exc:
            assert False, exc
        assert policy is not None

    def test_gradient_ascent(self):
        P = RockPaperScissors()
        policy = [GradientAscent.create_from_game(P, i) for i in P.I]
        for policy_i in policy:
            for a_i in P.A[policy_i.i]:
                assert(np.isclose(policy_i.policy_i(a_i), 1 / len(P.A[policy_i.i])))
        
        a = tuple([policy_i() for policy_i in policy])
        for policy_i in policy:
            policy_i.update(a)
            assert(policy_i.t == 2)
            assert(np.isclose(sum([v for v in policy_i.policy_i.p.values()]), 1))

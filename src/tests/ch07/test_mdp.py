import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np

from typing import Any

from problems.HexWorldMDP import ThreeTileStraightLineHexWorld, three_tile_init_policy, action_to_idx


class TestMDP():
    P = ThreeTileStraightLineHexWorld

    @staticmethod
    def U1(s: Any) -> float:
        return 0.0

    @staticmethod
    def U2(s: Any) -> float:
        if s == 0:
            return 1.0
        elif s == 1:
            return 2.0
        elif s == 2:
            return 3.0
        else:  # s == 3
            return 4.0

    U1_vec = np.zeros(4)
    U2_vec = np.array([1.0, 2.0, 3.0, 4.0])
    correct_policy_utility = np.array([1.425, 2.128, 10.0, 0.0])

    def test_lookahead(self):
        assert np.isclose(np.abs(self.P.lookahead(TestMDP.U1, s=0, a=action_to_idx["east"]) - (-0.3)), 0, atol=1e-10)
        assert np.isclose(np.abs(self.P.lookahead(TestMDP.U1_vec, s=0, a=action_to_idx["east"]) - (-0.3)), 0, atol=1e-10)
        assert self.P.lookahead(TestMDP.U2, s=0, a=action_to_idx["east"]) == 1.23
        assert self.P.lookahead(TestMDP.U2_vec, s=0, a=action_to_idx["east"]) == 1.23

    def test_policy_evaluation(self, tol=1e-3):
        utility = self.P.policy_evaluation(three_tile_init_policy)
        assert np.all(np.abs(self.correct_policy_utility - utility) < tol)

    def test_iterative_policy_evaluation(self, tol=1e-3):
        utility = self.P.iterative_policy_evaluation(three_tile_init_policy, k_max=100)
        assert np.all(np.abs(self.correct_policy_utility - utility) < tol)

    def test_greedy(self):
        assert self.P.greedy(TestMDP.U2, s=0) == (action_to_idx["east"], 1.23)
        assert self.P.greedy(TestMDP.U2_vec, s=0) == (action_to_idx["east"], 1.23)

    def test_backup(self):
        assert self.P.backup(TestMDP.U2, s=0) == 1.23
        assert self.P.backup(TestMDP.U2_vec, s=0) == 1.23

    def test_randstep(self, tol=1e-2):
        count = 0
        n_trials = 100000
        for _ in range(n_trials):
            possible_results = [(0, -0.3), (1, -0.3)]
            state, reward = self.P.randstep(s=0, a=action_to_idx["east"])
            result = (state, round(reward, ndigits=1))
            assert result in possible_results
            if result == possible_results[0]:
                count += 1
        assert np.abs(0.3 - (count / n_trials)) < tol

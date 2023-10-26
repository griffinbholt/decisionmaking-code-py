import sys; sys.path.append('./code/'); sys.path.append('../../')

import numpy as np
from typing import Any, Callable

from ch07 import MDP, ExactSolutionMethod, PolicyIteration, ValueIteration, GaussSeidelValueIteration, LinearProgramFormulation
from ThreeTileStraightlineHexworld import gamma, S, A, T, R, TR, policy

class TestExactSolutionMethods():
    P = MDP(gamma, S, A, T, R, TR)

    def test_policy_iteration(self):
        def initial_policy(s): return "west"
        policy_iteration = PolicyIteration(initial_policy, k_max=10)
        policy = self.run_solve(policy_iteration)
        self.assert_all_east(policy)

    def test_value_iteration(self):  # See Exercise 7.6
        value_iteration = ValueIteration(k_max=2)
        policy = self.run_solve(value_iteration)
        assert(np.abs(policy.U[0] - (-0.57)) < 1e-10)
        assert(np.abs(policy.U[1] - (5.919)) < 1e-10)
        assert(np.abs(policy.U[2] - (10)) < 1e-10)

    def test_gauss_seidel_valit(self):
        gs_value_iteration = GaussSeidelValueIteration(k_max=10)
        policy = self.run_solve(gs_value_iteration)
        self.assert_all_east(policy)

    def test_linear_program_form(self):
        lp_solver = LinearProgramFormulation()
        policy = self.run_solve(lp_solver)
        self.assert_all_east(policy)

    def run_solve(self, M: ExactSolutionMethod) -> Callable[[Any], Any]:
        return M.solve(self.P)

    def assert_all_east(self, policy: Callable[[Any], Any]):
        for state in self.P.S:
            assert(policy(state) == "east")

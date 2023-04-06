import sys; sys.path.append('./code/'); sys.path.append('../../')

from typing import Any, Callable

from ch07 import MDP, ExactSolutionMethod, PolicyIteration, ValueIteration, GaussSeidelValueIteration, LinearProgramFormulation


class TestExactSolutionMethods():  # TODO
    def test_policy_iteration(self):
        pass

    def test_value_iteration(self):
        pass

    def test_gauss_seidel_valit(self):
        pass

    def test_linear_program_form(self):
        pass

    def run_solve(self, M: ExactSolutionMethod) -> Callable[[Any], Any]:
        pass

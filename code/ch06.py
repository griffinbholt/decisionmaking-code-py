import numpy as np

from ch02 import Variable, Assignment, BayesianNetwork, assignments
from ch03 import DiscreteInferenceMethod


class SimpleProblem():
    def __init__(self, 
                 bn: BayesianNetwork, 
                 chance_vars: list[Variable], 
                 decision_vars: list[Variable], 
                 utility_vars: list[Variable], 
                 utilities: dict[str, np.ndarray]):
        self.bn = bn
        self.chance_vars = chance_vars
        self.decision_vars = decision_vars
        self.utility_vars = utility_vars
        self.utilities = utilities

    def solve(self, evidence: Assignment, M: DiscreteInferenceMethod) -> tuple[Assignment, float]:
        query = [var.name for var in self.utility_vars]
        def U(a: Assignment): return np.sum([self.utilities[uname][a[uname]] for uname in query])
        best_a, best_u = None, -np.inf
        for assignment in assignments(self.decision_vars):
            evidence = Assignment(evidence | assignment)
            phi = M.infer(self.bn, query, evidence)
            u = np.sum([p*U(a) for a, p in phi.table.items()])
            if u > best_u:
                best_a, best_u = assignment, u
        return best_a, best_u

    def value_of_information(self, query: list[str], evidence: Assignment, M: DiscreteInferenceMethod) -> float:
        phi = M.infer(self.bn, query, evidence)
        voi = -(self.solve(evidence, M)[1])
        query_vars = [var for var in self.chance_vars if var.name in query]
        for o_prime in assignments(query_vars):
            o_o_prime = Assignment(evidence | o_prime)
            p = phi.table[o_prime]
            voi += p*(self.solve(o_o_prime, M)[1])
        return voi
    
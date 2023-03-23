import networkx as nx
import numpy as np

from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal

from ch2 import Assignment, Factor, FactorTable, BayesianNetwork
from ch2 import marginalize, condition_multiple

class InferenceMethod(ABC):
    @abstractmethod
    def infer(self, *args, **kwargs):
        pass

class DiscreteInferenceMethod(InferenceMethod):
    @abstractmethod
    def infer(self, bn: BayesianNetwork, query: list[str], evidence: Assignment) -> Factor:
        pass

class ExactInference(DiscreteInferenceMethod):
    def infer(self, bn: BayesianNetwork, query: list[str], evidence: Assignment) -> Factor:
        phi = Factor.prod(bn.factors)
        phi = condition_multiple(phi, evidence)
        for name in (set(phi.variable_names) - set(query)):
            phi = marginalize(phi, name)
        phi.normalize()
        return phi

class VariableElimination(DiscreteInferenceMethod):
    def __init__(self, ordering: list[int]):
        self.ordering = ordering

    def infer(self, bn: BayesianNetwork, query: list[str], evidence: Assignment) -> Factor:
        factors = [condition_multiple(phi, evidence) for phi in bn.factors]
        for i in self.ordering:
            name = bn.variables[i].name
            if name not in query:
                indices = [phi for phi in bn.factors if phi.in_scope(name)]
                if len(indices) != 0:
                    phi = Factor.prod([factors[j] for j in indices])
                    for j in sorted(indices, reverse=True):
                        del factors[j]
                    phi = marginalize(phi, name)
                    factors.append(phi)
        phi = Factor.prod(factors)
        phi.normalize()
        return phi

class DirectSampling(DiscreteInferenceMethod):
    def __init__(self, m):
        self.m = m

    def infer(self, bn: BayesianNetwork, query: list[str], evidence: Assignment) -> Factor:
        table = FactorTable()
        for _ in range(self.m):
            a = bn.sample()
            print(a)
            if all(a[k] == v for k,v in evidence.items()):
                b = a.select(query)
                table[b] = table.get(b, default_val=0.0) + 1
        variables = [var for var in bn.variables if var.name in query]
        phi = Factor(variables, table)
        phi.normalize()
        return phi

class LikelihoodWeightedSampling(DiscreteInferenceMethod):
    def __init__(self, m):
        self.m = m

    def infer(self, bn: BayesianNetwork, query: list[str], evidence: Assignment) -> Factor:
        table = FactorTable()
        ordering = list(nx.topological_sort(bn.graph))
        for _ in range(self.m):
            a, w = Assignment(), 1.0
            for j in ordering:
                name, phi = bn.variables[j].name, bn.factors[j]
                if name in evidence:
                    a[name] = evidence[name]
                    w *= phi.table[a.select(phi.variable_names)]
                else:
                    a[name] = condition_multiple(phi, a).sample()[name]
            b = a.select(query)
            table[b] = table.get(b, default_val=0.0) + w 
        variables = [var for var in bn.variables if var.name in query]
        phi = Factor(variables, table)
        phi.normalize()
        return phi

class GibbsSampling(DiscreteInferenceMethod):
    def __init__(self, m_samples: int, m_burnin: int, m_skip: int, ordering: list[int]):
        self.m_samples = m_samples
        self.m_burnin = m_burnin
        self.m_skip = m_skip
        self.ordering = ordering 

    def infer(self, bn: BayesianNetwork, query: list[str], evidence: Assignment) -> Factor:
        table = FactorTable()
        a = Assignment(bn.sample() | evidence)
        a.gibbs_sample(bn, evidence, self.ordering, self.m_burnin)
        for i in range(self.m_samples):
            self.gibbs_sample(a, bn, evidence, self.ordering, self.m_skip)
            b = a.select(query)
            table[b] = table.get(b, default_val=0) + 1
        variables = [var for var in bn.variables if var.name in query]
        phi = Factor(variables, table)
        phi.normalize()
        return phi

    @staticmethod
    def gibbs_sample(a: Assignment, bn: BayesianNetwork, evidence: Assignment, ordering: list[int], m: int):
        for _ in range(m):
            self.update_gibbs_sample(a, bn, evidence, ordering)

    @staticmethod
    def update_gibbs_sample(a: Assignment, bn: BayesianNetwork, evidence: Assignment, ordering: list[int]):
        for i in ordering:
            name = bn.variables[i].name
            if name not in evidence:
                b = self.blanket(bn, a, i)
                a[name] = b.sample()[name]

    @staticmethod
    def blanket(bn: BayesianNetwork, a: Assignment, i: int) -> Factor:
        name = bn.variables[i].name
        value = a[name]
        a_prime = a.copy()
        del a_prime[name]
        factors = [phi for phi in bn.factors if phi.in_scope(name)]
        phi = Factor.prod([condition_multiple(factor, a_prime) for factor in factors])
        phi.normalize()
        return phi

class MultivariateGaussianInference(InferenceMethod):
    """
        D: multivariate_normal object defined by scipy.stats
        query: NumPy array of integers specifying the query variables
        evidencevars: NumPy array of integers specifying the evidence variables
        evidence: NumPy array containing the values of the evidence variables
    """
    def infer(self, D: multivariate_normal, query: np.ndarray, evidence_vars: np.ndarray, evidence: np.ndarray) -> multivariate_normal:
        mu, Sigma = D.mean, D.cov
        b, mu_a, mu_b = evidence, mu[query], mu[evidence_vars]
        A = Sigma[query][:, query]
        B = Sigma[evidence_vars][:, evidence_vars]
        C = Sigma[query][:, evidence_vars]
        mu = mu_a + (C @ np.linalg.solve(B, b - mu_b))
        Sigma = A - (C @ (np.linalg.inv(B) @ C.T))
        return multivariate_normal(mu, Sigma)
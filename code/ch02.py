import itertools
import networkx as nx
import numpy as np

from functools import reduce


class Variable():
    def __init__(self, name: str, r: int):
        self.name = name
        self.r = r  # number of possible values

    def __str__(self):
        return "(" + self.name + ", " + str(self.r) + ")"


"""
Assignment: A mapping from variable names (str) to integer values (int)
"""
class Assignment(dict[str, int]):
    def select(self, varnames: list[str]) -> 'Assignment':
        return Assignment({n: dict.__getitem__(self, n) for n in varnames})

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.items())))

    def copy(self) -> 'Assignment':
        result = Assignment() 
        result.update(self)
        return result


"""
FactorTable: A mapping from assignments (Assignment) to values (float)
"""
class FactorTable(dict[Assignment, float]):
    def get(self, key: Assignment, default_val: float):
        return dict.__getitem__(self, key) if key in self.keys() else default_val

    def __str__(self) -> str:
        table_str = ""
        for key in self.keys():
            table_str += str(key) + ": " + str(dict.__getitem__(self, key)) + "\n"
        table_str = table_str[:-1]
        return table_str


class Factor():
    def __init__(self, variables: list[Variable], table: FactorTable):
        self.variables = variables
        self.table = table        
        self.variable_names = [var.name for var in variables]

    def __str__(self) -> str:
        factor_str = "Variables: "
        factor_str += str([str(var) for var in self.variables])
        factor_str += '\n'
        factor_str += str(self.table)
        return factor_str

    def normalize(self):
        z = np.sum([self.table[a] for a in self.table])
        for a, p in self.table.items():
            self.table[a] = p/z

    def __mul__(self, other: 'Factor') -> 'Factor':
        other_only = list(set(other.variables) - set(self.variables))
        table = FactorTable()
        for self_a, self_p in self.table.items():
            for a in assignments(other_only):
                a = Assignment(self_a | a)
                other_a = a.select(other.variable_names)
                table[a] = self_p * other.table.get(other_a, default_val=0.0)
        variables = self.variables + other_only
        return Factor(variables, table)

    def in_scope(self, name: str) -> bool:
        return any([name == var.name for var in self.variables])

    def sample(self) -> Assignment:
        total, p, w = 0.0, np.random.rand(), sum(self.table.values())
        for a, v in self.table.items():
            total += v/w
            if total >= p:
                return a 
        return Assignment()

    @staticmethod
    def prod(factors: list['Factor']) -> 'Factor':
        return reduce(lambda phi_1, phi_2: phi_1 * phi_2, factors)


def assignments(variables: list[Variable]) -> list[Assignment]:
    names = [var.name for var in variables]
    return [Assignment(zip(names, values)) for values in itertools.product(*[[i for i in range(var.r)] for var in variables])]

# Note: Although `marginalize`, `condition_single`, and `condition_multiple` are
# all defined in Chapter 3, the `BayesianNetwork.sample()` function depends on `condition_multiple`.
# To import `condition_multiple` would create a circular import issue. Thus, these functions are included
# in the ch02 code module, but can still be displayed in Chapter 3 of the textbook.

def marginalize(phi: Factor, name: str) -> Factor:
    table = FactorTable()
    for a, p in phi.table.items():
        a_prime = a.copy()
        del a_prime[name]
        table[a_prime] = table.get(a_prime, default_val=0.0) + p
    variables = [var for var in phi.variables if var.name is not name]
    return Factor(variables, table)


def condition_single(phi: Factor, name: str, value: int) -> Factor:
    if not phi.in_scope(name):
        return phi
    table = FactorTable()
    for a, p in phi.table.items():
        if a[name] == value:
            a_prime = a.copy()
            del a_prime[name]
            table[a_prime] = p
    variables = [var for var in phi.variables if var.name is not name]
    return Factor(variables, table)


def condition_multiple(phi: Factor, evidence: Assignment) -> Factor:
    for name, value in evidence.items():
        phi = condition_single(phi, name, value)
    return phi


class BayesianNetwork():
    def __init__(self, variables: list[Variable], factors: list[Factor], graph: nx.DiGraph):
        self.variables = variables
        self.factors = factors
        self.graph = graph

    def probability(self, assignment: Assignment) -> float:
        def subassignment(phi: Factor): return assignment.select(phi.variable_names)
        def prob(phi: Factor): return phi.table.get(subassignment(phi), default_val=0.0)
        return np.prod([prob(phi) for phi in self.factors])

    def sample(self) -> Assignment:
        a = Assignment()
        for i in list(nx.topological_sort(self.graph)):
            name, phi = self.variables[i].name, self.factors[i]
            a[name] = (condition_multiple(phi, a).sample())[name]
        return a

import networkx as nx
import sys; sys.path.append('../')

from ch02 import Variable, Assignment, FactorTable, Factor, BayesianNetwork


def example_2_3():
    """Example 2.3: Constructing a discrete factor"""
    X = Variable("x", 2)
    Y = Variable("y", 2)
    Z = Variable("z", 2)
    phi = Factor([X, Y, Z], FactorTable({
        Assignment({"x": 0, "y": 0, "z": 0}): 0.08, Assignment({"x": 0, "y": 0, "z": 1}): 0.31,
        Assignment({"x": 0, "y": 1, "z": 0}): 0.09, Assignment({"x": 0, "y": 1, "z": 1}): 0.37,
        Assignment({"x": 1, "y": 0, "z": 0}): 0.01, Assignment({"x": 1, "y": 0, "z": 1}): 0.05,
        Assignment({"x": 1, "y": 1, "z": 0}): 0.02, Assignment({"x": 1, "y": 1, "z": 1}): 0.07}))
    return phi


def example_2_5():
    """
    Example 2.5: A Bayesian network representing a satellite-monitoring problem.
    Associated with each node is a conditional probability distribution.
    """
    B = Variable("b", 2)  # battery failure
    S = Variable("s", 2)  # solar panel failure
    E = Variable("e", 2)  # electrical system failure
    D = Variable("d", 2)  # trajectory deviation
    C = Variable("c", 2)  # communication loss
    variables = [B, S, E, D, C]
    factors = [
        Factor([B], FactorTable({Assignment({"b": 0}): 0.99, Assignment({"b": 1}): 0.01})),
        Factor([S], FactorTable({Assignment({"s": 0}): 0.98, Assignment({"s": 1}): 0.02})),
        Factor([E, B, S], FactorTable({
            Assignment({"e": 0, "b": 0, "s": 0}): 0.90, Assignment({"e": 0, "b": 0, "s": 1}): 0.04,
            Assignment({"e": 0, "b": 1, "s": 0}): 0.05, Assignment({"e": 0, "b": 1, "s": 1}): 0.01,
            Assignment({"e": 1, "b": 0, "s": 0}): 0.10, Assignment({"e": 1, "b": 0, "s": 1}): 0.96,
            Assignment({"e": 1, "b": 1, "s": 0}): 0.95, Assignment({"e": 1, "b": 1, "s": 1}): 0.99})),
        Factor([D, E], FactorTable({
            Assignment({"d": 0, "e": 0}): 0.96, Assignment({"d": 0, "e": 1}): 0.03,
            Assignment({"d": 1, "e": 0}): 0.04, Assignment({"d": 1, "e": 1}): 0.97})),
        Factor([C, E], FactorTable({
            Assignment({"c": 0, "e": 0}): 0.98, Assignment({"c": 0, "e": 1}): 0.01,
            Assignment({"c": 1, "e": 0}): 0.02, Assignment({"c": 1, "e": 1}): 0.99}))
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from([(0, 2), (1, 2), (2, 3), (2, 4)])
    bn = BayesianNetwork(variables, factors, graph)
    return bn

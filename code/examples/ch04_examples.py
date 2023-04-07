import networkx as nx
import numpy as np
import sys; sys.path.append('../')

from ch02 import Variable
from ch04 import statistics, prior


def example_4_1():
    """
    Example 4.1:
    Using the statistics function for extracting the statistics from a data set. 
    Bayesian parameter learning can be used to avoid `nan` values, but we must specify a prior.

    Note: `np.ravel_multi_index` indexes the parental instantiations differently than the textbook's `sub2ind`
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(3))
    graph.add_edges_from([(0, 1), (2, 1)])
    variables = [Variable("A", 2), Variable("B", 2), Variable("C", 2)]
    data = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [1, 1, 1, 1]])
    M = statistics(variables, graph, data)
    mle = [M_i / M_i.sum(axis=1, keepdims=True) for M_i in M]
    return variables, graph, M, mle


def example_4_2():
    """
    Example 4.2:
    Computing the posterior parameters in a Bayesian network. 
    Note that unlike example 4.1, here we do not have `nan` values.

    Note: `np.ravel_multi_index` indexes the parental instantiations differently than the textbook's `sub2ind`
    """
    variables, graph, M, _ = example_4_1()
    alpha = prior(variables, graph)
    posterior_params = [alpha[i] + M[i] for i in range(len(variables))]
    return posterior_params

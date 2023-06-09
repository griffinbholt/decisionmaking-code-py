import networkx as nx
import numpy as np
import sys; sys.path.append('../')

from ch02 import Variable
from ch04 import statistics


def exercise_4_4():
    """
    Exercise 4.4: MLEs of conditional distribution parameters for Bayesian network

    Note: `np.ravel_multi_index` indexes the parental instantiations differently than the textbook's `sub2ind`
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 3), (3, 1), (2, 1)])
    variables = [Variable("X1", 2), Variable("X2", 2), Variable("X3", 2), Variable("X4", 3)]
    data = np.array([[0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                     [1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
                     [1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
                     [2, 1, 0, 0, 0, 2, 2, 0, 0, 0]])
    M = statistics(variables, graph, data)
    mle = [M_i / M_i.sum(axis=1, keepdims=True) for M_i in M]
    print(M)
    print(mle)

import networkx as nx
import numpy as np
import sys; sys.path.append('../')

from ch2 import Variable
from ch4 import statistics, prior

# Example 4.1: Extracting statistics from a data set
# Note: np.ravel_multi_index indexes the parental instantiations differently than sub2ind 
def example_4_1():
    graph = nx.DiGraph()
    graph.add_nodes_from(range(3))
    graph.add_edges_from([(0, 1), (2, 1)])
    variables = [Variable("A", 2), Variable("B", 2), Variable("C", 2)]
    data = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [1, 1, 1, 1]])
    M = statistics(variables, graph, data)
    mle = [M_i / M_i.sum(axis=1, keepdims=True) for M_i in M]
    return variables, graph, M, mle

# Example 4.2: Computing the posterior parameters in a Bayesian network
# Note: np.ravel_multi_index indexes the parental instantiations differently than sub2ind 
def example_4_2():
    variables, graph, M, _ = example_4_1()
    alpha = prior(variables, graph)
    posterior_params = [alpha[i] + M[i] for i in range(len(variables))]
    return posterior_params


import networkx as nx
import numpy as np
import pandas as pd
import pytest
import sys; sys.path.append('./code/')

from ch02 import Variable
from ch05 import bayesian_score

def test_bayesian_scoring(tol=1e-15):
    data = pd.read_csv("./code/tests/ch05/example/example.csv").to_numpy().T - 1
    graph = nx.DiGraph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from([(0, 1), (2, 3), (4, 5)])
    variables = [Variable("p1", 3), Variable("c1", 3), Variable("p2", 3), Variable("c2", 3), Variable("p3", 3), Variable("c3", 3)]
    example_score = -132.02362143513778

    score = bayesian_score(variables, graph, data)

    assert np.abs(example_score - score) < tol
    
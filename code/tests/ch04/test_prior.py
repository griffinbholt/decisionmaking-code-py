import networkx as nx 
import numpy as np
import pytest
import sys; sys.path.append('./code/'); sys.path.append('../../')

from ch02 import Variable
from ch04 import prior

class TestPrior():
    def test(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(3))
        graph.add_edges_from([(0, 1), (2, 1)])
        variables = [Variable("A", 2), Variable("B", 2), Variable("C", 2)]
        alpha = prior(variables, graph)

                    # Value: 0  1    # Parents
        alpha_A = np.array( [1, 1] ) # None
        alpha_B = np.array([[1, 1],  # A = 0, C = 0
                            [1, 1],  # A = 0, C = 1
                            [1, 1],  # A = 1, C = 0
                            [1, 1]]) # A = 1, C = 1
        alpha_C = np.array( [1, 1] )

        assert np.all(alpha[0] == alpha_A)
        assert np.all(alpha[1] == alpha_B)
        assert np.all(alpha[2] == alpha_C)
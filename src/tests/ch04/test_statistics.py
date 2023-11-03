import networkx as nx
import numpy as np
import sys; sys.path.append('./src/'); sys.path.append('../../')

from ch02 import Variable
from ch04 import statistics


class TestStatistics():
    def test(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(3))
        graph.add_edges_from([(0, 1), (2, 1)])
        variables = [Variable("A", 2), Variable("B", 2), Variable("C", 2)]
        data = np.array([[0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [1, 1, 1, 1]])
        M = statistics(variables, graph, data)

                # Value: 0  1     # Parents:
        M_A =  np.array([2, 2])   # None
        M_B = np.array([[0, 0],   # A = 0, C = 0
                        [2, 0],   # A = 0, C = 1
                        [0, 0],   # A = 1, C = 0
                        [0, 2]])  # A = 1, C = 1
        M_C =  np.array([0, 4])   # None

        assert np.all(M[0] == M_A)
        assert np.all(M[1] == M_B)
        assert np.all(M[2] == M_C)

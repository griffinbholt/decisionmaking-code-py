import itertools
import networkx as nx
import numpy as np

from abc import ABC, abstractmethod
from scipy.special import loggamma

from ch02 import Variable
from ch04 import statistics, prior


def bayesian_score_component(M: np.ndarray, alpha: np.ndarray) -> float:
    alpha_0 = np.sum(alpha, axis=1)
    p = np.sum(loggamma(alpha + M))
    p -= np.sum(loggamma(alpha))
    p += np.sum(loggamma(alpha_0))
    p -= np.sum(loggamma(alpha_0 + np.sum(M, axis=1)))
    return p


def bayesian_score(variables: list[Variable], graph: nx.DiGraph, data: np.ndarray) -> float:
    n = len(variables)
    M = statistics(variables, graph, data)
    alpha = prior(variables, graph)
    return np.sum([bayesian_score_component(M[i], alpha[i]) for i in range(n)])


class DirectedGraphSearchMethod(ABC):
    @abstractmethod
    def fit(self, variables: list[Variable], data: np.ndarray) -> nx.DiGraph:
        pass


class K2Search(DirectedGraphSearchMethod):
    def __init__(self, ordering: list[int]):
        self.ordering = ordering

    def fit(self, variables: list[Variable], data: np.ndarray) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(variables)))
        for k, i in enumerate(self.ordering[1:]):
            y = bayesian_score(variables, graph, data)
            while True:
                y_best, j_best = -np.inf, 0
                for j in self.ordering[:k]:
                    if not graph.has_edge(j, i):
                        graph.add_edge(j, i)
                        y_prime = bayesian_score(variables, graph, data)
                        if y_prime > y_best:
                            y_best, j_best = y_prime, j
                        graph.remove_edge(j, i)
                if y_best > y:
                    y = y_best
                    graph.add_edge(j_best, i)
                else:
                    break
        return graph


class LocalDirectedGraphSearch(DirectedGraphSearchMethod):
    def __init__(self, initial_graph: nx.DiGraph, k_max: int):
        self.initial_graph = initial_graph
        self.k_max = k_max

    def fit(self, variables: list[Variable], data: np.ndarray) -> nx.DiGraph:
        graph = self.initial_graph
        y = bayesian_score(variables, graph, data)
        for k in range(self.k_max):
            graph_prime = self.rand_graph_neighbor(graph)
            if len(list(nx.simple_cycles(graph_prime))) == 0:
                y_prime = bayesian_score(variables, graph_prime, data) 
            else:
                y_prime = -np.inf
            if y_prime > y:
                y, graph = y_prime, graph_prime
        return graph

    @staticmethod
    def rand_graph_neighbor(graph: nx.DiGraph) -> nx.DiGraph:
        n = graph.number_of_nodes()
        i = np.random.randint(low=0, high=n)
        j = (i + np.random.randint(low=1, high=n) - 1) % n
        graph_prime = graph.copy()
        if graph.has_edge(i, j):
            graph_prime.remove_edge(i, j)
        else:
            graph_prime.add_edge(i, j)
        return graph_prime


def are_markov_equivalent(G: nx.DiGraph, H: nx.DiGraph) -> bool:
    if ((G.number_of_nodes() != H.number_of_nodes()) or 
        (G.number_of_edges() != H.number_of_edges()) or 
        (not all([(H.has_edge(e[0], e[1]) or H.has_edge(e[1], e[0])) for e in G.edges]))):
        return False
    for (I, J) in [(G, H), (H, G)]:
        for c in range(I.number_of_nodes()):
            parents = list(I.predecessors(c))
            for a, b in itertools.combinations(parents, 2):
                if (not I.has_edge(a, b)) and (not I.has_edge(b, a)) and (not (J.has_edge(a, c) and J.has_edge(b, c))):
                    return False
    return True
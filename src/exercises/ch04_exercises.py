import networkx as nx
import numpy as np
import sys; sys.path.append('../')

from scipy.stats import beta, multivariate_normal

from ch02 import Variable
from ch03 import MultivariateGaussianInference
from ch04 import statistics


def exercise_4_1():
    """Exercise 4.1: Probability of next basket"""
    # Uniform prior
    distrib = beta(1, 1)

    # Two Baskets, One Miss
    a, b = distrib.args
    distrib = beta(a + 2, b + 1)

    print("Probability of making next basket: ", distrib.mean())


def exercise_4_3():
    """Exercise 4.3: MLE for Censored Data"""
    n = 3  # number of fully observed measurements
    m = 2  # number of censored measurements

    t_known = [132, 42, 89]
    t_unknown_min = 200

    lam = n / (np.sum(t_known) + m*t_unknown_min)

    print("MLE for Lambda: ", lam)


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


def exercise_4_5():
    """Exercise 4.5: Estimating Probability for Biased Coin"""
    n = 1  # number of successes
    m = 1  # number of trials

    a, b = 1 + n, 1 + m - n  # posterior distribution

    print("MLE: phi = ", float(n/m))
    print("MAP: phi = ", float((a - 1)/(a + b - 2)))
    print("Posterior Distribution Mean: ", a / (a + b))


def exercise_4_6():
    """Exercise 4.6: Data Imputation for Gaussian Distribution"""
                        # X1    X2
    data = np.array([[   0.5,  1.0],
                     [np.nan,  0.3],
                     [  -0.6, -0.3],
                     [   0.1,  0.2]])
    unknown_idx = (1, 0)
    known_idx = (1, 1)

    neighbors = np.array([data[i] for i in range(len(data)) if i != unknown_idx[0]])

    # Marginal Mode Imputation
    marginal_mode = neighbors[:, 0].mean()
    print("Marginal Mode Imputed Value: ", marginal_mode)

    # Nearest-Neighbor Imputation
    nn_value = neighbors[np.argmin(np.abs(data[known_idx] - neighbors[:, known_idx[1]])), unknown_idx[1]]
    print("Nearest Neighbor Imputed Value: ", nn_value)


def exercise_4_7():
    """
    Exercise 4.7: Posterior Mode Imputation for Gaussian Distribution
    
    According to Section 4.4.1, we can use Algorithm 3.11 (Multivariate Gaussian
    Inference) to infer the posterior mode distribution.
    """
    # Joint Gaussian Distribution fit to X1, X2 data
    mean = np.array([5, 2])
    cov = np.array([[4, 1],
                    [1, 2]])
    D = multivariate_normal(mean, cov)

    # Query: X1 (Missing Data)
    query = np.array([0])

    # Evidence: X2 = 1.5 (Known Data)
    evidence_vars = np.array([1])
    evidence = np.array([1.5])

    M = MultivariateGaussianInference()
    cond_distrib = M.infer(D, query, evidence_vars, evidence)
    print("Posterior Distribution Mean: mu = ", cond_distrib.mean[0])
    print("Posterior Distirbution Variance: sigma^2 = ", cond_distrib.cov[0, 0])

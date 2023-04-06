import numpy as np
import sys; sys.path.append('../')
from scipy.stats import multivariate_normal

from ch02 import Variable, Assignment, FactorTable, Factor, marginalize, condition_single
from ch03 import DirectSampling, LikelihoodWeightedSampling, MultivariateGaussianInference
from ch02_examples import example_2_3, example_2_5


# Example 3.1: The Factor Product
def example_3_1():
    X = Variable("x", 2)
    Y = Variable("y", 2)
    Z = Variable("z", 2)
    phi_1 = Factor([X, Y], FactorTable({
        Assignment({"x": 0, "y": 0}): 0.3, Assignment({"x": 0, "y": 1}): 0.4,
        Assignment({"x": 1, "y": 0}): 0.2, Assignment({"x": 1, "y": 1}): 0.1}))
    phi_2 = Factor([Y, Z], FactorTable({
        Assignment({"y": 0, "z": 0}): 0.2, Assignment({"y": 0, "z": 1}): 0.0,
        Assignment({"y": 1, "z": 0}): 0.3, Assignment({"y": 1, "z": 1}): 0.5}))
    phi_3 = phi_1 * phi_2
    return phi_3


# Example 3.2: Factor Marginalization
def example_3_2():
    phi_1 = example_2_3()
    phi_2 = marginalize(phi_1, "y")
    return phi_2


# Example 3.3: Factor Conditioning
def example_3_3():
    phi_1 = example_2_3()
    phi_2 = condition_single(phi_1, name="y", value=1)
    return phi_2


# Example 3.5: Direct Sampling
# Note: The samples drawn here are drawn from the factors in Ex. 2.5,
# and will be different than those in the textbook
# I also changed the number of samples from 10 to 1000 to make it more interesting
def example_3_5():
    bn = example_2_5()
    query = ["b"]
    evidence = Assignment({"d": 1, "c": 1})
    M = DirectSampling(m=1000)
    phi = M.infer(bn, query, evidence)
    return phi


# Example 3.6: Likelihood Weighted Sampling
# Note: The samples drawn here are drawn from the factors in Ex. 2.5,
# and will be different than those in the textbook.
# I also changed the number of samples from 5 to 1000 to make it more interesting
def example_3_6():
    bn = example_2_5()
    query = ["b"]
    evidence = Assignment({"d": 1, "c": 1})
    M = LikelihoodWeightedSampling(m=1000)
    phi = M.infer(bn, query, evidence)
    return phi


# Example 3.7: Multivariate Gaussian Inference
def example_3_7():
    # Joint Distribution
    mean = np.array([0, 1])
    cov = np.array([[3, 1], [1, 2]])
    D = multivariate_normal(mean, cov)

    # Query: x_1
    query = np.array([0])

    # Evidence: x_2 = 2.0
    evidence_vars = np.array([1])
    evidence = np.array([2.0])

    M = MultivariateGaussianInference()
    cond_distrib = M.infer(D, query, evidence_vars, evidence)
    return cond_distrib

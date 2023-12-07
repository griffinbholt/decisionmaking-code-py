import sys; sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from ch07 import LinearQuadraticProblem
from problems.HexWorldMDP import StraightLineHexWorld


def example_7_2():
    """
    Example 7.2: The effect of the dicsount factor on convergence of
    value iteration. In each case, value iteration was run until the Bellman
    residual was less than 1.
    """
    P = StraightLineHexWorld

    def value_iteration(P, gamma):
        P.gamma = gamma
        U_all = np.zeros((1, len(P.S)))
        U_curr = np.zeros(len(P.S))
        bellman_residual = np.infty
        while bellman_residual >= 1:
            U_curr = np.array([P.backup(U_curr, s) for s in P.S])
            bellman_residual = np.linalg.norm(U_curr - U_all[-1], ord=np.infty)
            U_all = np.vstack([U_all, U_curr[np.newaxis]])
        return U_all
    
    # Value Iteration w/ high discount factor:
    U_hi = value_iteration(P, gamma=0.9)

    # Value Iteration w/ low discount factor:
    U_lo = value_iteration(P, gamma=0.5)

    # Display the results
    _, ax = plt.subplots(1, 2)
    ax[0].matshow(U_hi[:, :-1])
    ax[0].set_title("gamma = 0.9")
    ax[0].set_anchor('N')
    ax[1].matshow(U_lo[:, :-1])
    ax[1].set_title("gamma = 0.5")
    ax[1].set_anchor('N')
    plt.show()


def example_7_3():
    """
    Example 7.3: The effect of the state ordering on convergence of asynchronous
    value iteration. In this, evaluating right to left allows convergence to occur
    in far fewer iterations.
    """
    P = StraightLineHexWorld

    def async_value_iteration(P, state_ordering):
        U_all = np.zeros((1, len(P.S)))
        U_curr = np.zeros(len(P.S))
        bellman_residual = np.infty
        while bellman_residual >= 1:
            for s in state_ordering:
                U_curr[s] = P.backup(U_curr, s)
            bellman_residual = np.linalg.norm(U_curr - U_all[-1], ord=np.infty)
            U_all = np.vstack([U_all, U_curr[np.newaxis]])
        return U_all
    
    # Async Value Iteration – Left to Right:
    U_lr = async_value_iteration(StraightLineHexWorld,
                                 state_ordering=P.S)

    # Async Value Iteration – Right to Left:
    U_rl = async_value_iteration(StraightLineHexWorld,
                                 state_ordering=list(reversed(P.S)))

    # Display the results
    _, ax = plt.subplots(1, 2)
    ax[0].matshow(U_lr[:, :-1])
    ax[0].set_title("left to right")
    ax[0].set_anchor('N')
    ax[1].matshow(U_rl[:, :-1])
    ax[1].set_title("right to left")
    ax[1].set_anchor('N')
    plt.show()


def example_7_4():
    """Example 7.4: Solving a finite horizon MDP with a linear transition function and quadratic reward"""
    t_step = 1
    Ts = np.array([[1, t_step], [0, 1]])
    Ta = np.array([[0.5*(t_step**2)], [t_step]])
    Rs = -np.eye(2)
    Ra = -np.array([[0.5]])
    h_max = 5

    lqp = LinearQuadraticProblem(Ts, Ta, Rs, Ra, h_max)
    opt_policies = lqp.solve()

    s = np.array([-10, 0])
    for i in range(h_max):
        a = opt_policies[i](s)
        s = (Ts @ s) + (Ta @ a)
        print(s)

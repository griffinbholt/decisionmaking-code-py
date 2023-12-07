import sys; sys.path.append('../')

import numpy as np

from ch07 import MDP
from ch09 import MonteCarloTreeSearch

def exercise_9_9():
    """Exercise 9.9: Monte Carle Tree Search Traversal Action"""
    P = MDP(
        gamma=None,      # Not needed
        S=np.arange(2),  # S = [0, 1]
        A=np.arange(2),  # A = [0, 1]
        T=None,          # Not needed
        R=None,          # Not needed
        TR=None,         # None
    )
    Q = np.array([[10, -5],
                  [12, 10]])
    N = np.array([[27, 4],
                  [32, 18]])
    M = MonteCarloTreeSearch(
        P=P,
        N=N,
        Q=Q,
        d=None,  # Not needed
        m=None,  # Not needed
        c=None,  # To be set later
        U=None,  # Not needed
    )

    for c in [10, 20]:
        M.c = c
        Ns = np.sum(N, axis=1)
        UCB1 = np.array([[M.ucb1(s, a, Ns[s]) for a in P.A] for s in P.S])
        actions = np.argmax(UCB1, axis=1)
        
        # Report results
        header = '| {:>2} | {:>10} | {:>10} | {:>18} |'.format("s", "UCB(s, a1)", "UCB(s, a2)", "argmax_a UCB(s, a)")
        border = '-' * len(header)
        print("c = " + str(c) + ":")
        print(border)
        print(header)
        print(border)
        for s in P.S:
            print('| {:^2} | {:>10.3f} | {:>10.3f} | {:>18} |'.format("s" + str(s + 1), UCB1[s, 0], UCB1[s, 1], "a" + str(actions[s] + 1)))
        print(border, "\n")

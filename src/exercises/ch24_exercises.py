import sys; sys.path.append('../');

import numpy as np

from ch24 import SimpleGame, NashEquilibrium

def exercise_24_2():
    """
    Exercise 24.2: Example of Game w/ 2 agents, 2 actions, and 2 Nash equilibria
    involving deterministic policies
    """
    I = [0, 1]            # two agents
    A = [[0, 1], [0, 1]]  # 0: descend, 1: climb
    R_tensor = np.array([[[-4, -4], [0, -1]],
                         [[-1, 0], [-5, -5]]])
    def R(a, R_tensor=R_tensor): return R_tensor[a]
    P = SimpleGame(None, I, A, R)

    M = NashEquilibrium()
    policy = M.solve(P)

    for i, policy_i in enumerate(policy):
        print("Agent " + str(i) + ": ", policy_i.p)

# TODO - Exercise 24.5
# TODO - Exercise 24.6
# TODO - Exercise 24.7
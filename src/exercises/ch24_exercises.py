import sys; sys.path.append('../');

import numpy as np

from ch24 import SimpleGame, NashEquilibrium, IteratedBestResponse, FictitiousPlay
from problems.SimpleGames import RockPaperScissors, AlmostRockPaperScissors, TravelersDilemma

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

def exercise_24_5():
    """
    Exercise 24.5: Game w/ 2 agents, 2 actions, for which correlated
    equilibria cannot be expressed as a Nash equilibrium
    """
    I = [0, 1]            # two agents
    A = [[0, 1], [0, 1]]  # 0: dinner, 1: movie
    R_tensor = np.array([[[2, 1], [0.0, 0.0]],
                         [[0.0, 0.0], [1, 2]]])  # conflicting preference
    def R(a, R_tensor=R_tensor): return R_tensor[a]
    P = SimpleGame(None, I, A, R)  # Deciding on what kind of date

    print("Nash Equilibrium:")
    M = NashEquilibrium()
    policy = M.solve(P)
    for i, policy_i in enumerate(policy):
        print("Agent " + str(i) + ": ", policy_i.p)

def exercise_24_6():
    """
    Exercise 24.6: Iterated Best Response & Fictitious Play Not Converging
    """
    print("Iterated Best Response:")
    P = RockPaperScissors()
    M = IteratedBestResponse.create_from_game(P, k_max=10)
    policy = M.initial_policy
    header = '| {:<9} | {:<16} | {:<16} | {:^9} |'.format("Iteration", "Agent 1's Action", "Agent 2's Action", "Rewards")
    border = '-' * len(header)
    print(border)
    print(header)
    print(border)
    for k in range(M.k_max):
        policy = [P.best_response(policy, i) for i in P.I]
        actions = tuple([list(policy[i].p.keys())[0] for i in range(len(P.I))])
        rewards = P.R(actions)
        print('| {:<9} | {:<16} | {:<16} | {:^7} |'.format(str(k + 1), actions[0], actions[1],  '{:.1f}, {:.1f}'.format(*rewards)))
    print()

    print("Fictitious Play:")
    P = AlmostRockPaperScissors()
    policy = [FictitiousPlay.create_from_game(P, i) for i in P.I]
    print(border)
    print(header)
    print(border)
    for k in range(10):
        a = [policy_i() for policy_i in policy]
        for policy_i in policy:
            policy_i.update(a)
        actions = tuple([list(policy[i].policy_i.p.keys())[0] for i in P.I])
        rewards = P.R(actions)
        print('| {:<9} | {:<16} | {:<16} | {:^7} |'.format(str(k + 1), actions[0], actions[1],  '{:.1f}, {:.1f}'.format(*rewards)))
        
def exercise_24_7():
    """Exercise 24.7: Iterated Best Response on Traveler's Dilemma"""
    P = TravelersDilemma()
    M = IteratedBestResponse.create_from_game(P, k_max=100)
    policy = M.solve(P)
    for i, policy_i in enumerate(policy):
        print("Agent " + str(i) + ": ", policy_i.p)

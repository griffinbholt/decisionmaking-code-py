import sys; sys.path.append('../')

import numpy as np

from problems.HexWorldMDP import ThreeTileStraightLineHexWorld, three_tile_init_policy, idx_to_action
from ch07 import PolicyIteration, ValueIteration

def exercise_7_4():
    """Exercise 7.4: Computing Utility U(s), Policy pi(s), and Advantage A(s, a)
    from the Action Value Function Q(s, a)"""
    S = [1, 2, 3, 4, 5, 6]
    Q = np.array([[0.41, 0.46, 0.37, 0.37],
                  [0.50, 0.55, 0.46, 0.37],
                  [0.60, 0.50, 0.38, 0.44],
                  [0.41, 0.50, 0.33, 0.41],
                  [0.50, 0.60, 0.41, 0.39],
                  [0.71, 0.70, 0.61, 0.59]])  # (|S| x |A|)
    
    # Compute Utility, Policy, and Advantage
    U = np.max(Q, axis=1)  # U = max_a Q_(s, a)
    def policy(s): return np.argmax(Q, axis=1)[s]
    A = Q - U[:,np.newaxis]

    # Report results
    header = '| {:^2} | {:>4} | {:>5} | {:>7} | {:>7} | {:>7} | {:>7} |'.format("s", "U(s)", "pi(s)", "A(s,a1)", "A(s,a2)", "A(s,a3)", "A(s,a4)")
    border = '-' * len(header)
    print(header)
    print(border)
    for s in range(len(S)):
            print('| {:^2} | {:>4.2f} | {:^5} | {:>7.2f} | {:>7.2f} | {:>7.2f} | {:>7.2f} |'.format("s" + str(s + 1), U[s], "a" + str(policy(s) + 1), A[s, 0], A[s, 1], A[s, 2], A[s, 3]))
    print(border)

def exercise_7_5():
    """Exercise 7.5: Policy Iteration on Three-tile, Straight-line Hexworld"""
    P = ThreeTileStraightLineHexWorld

    M = PolicyIteration(initial_policy=three_tile_init_policy, k_max=1)
    resulting_policy = M.solve(P)

    print("pi(s_1) =", idx_to_action[resulting_policy(P.S[0])])
    print("pi(s_2) =", idx_to_action[resulting_policy(P.S[1])])
    print("pi(s_3) =", idx_to_action[resulting_policy(P.S[2])])

def exercise_7_6():
    """Exercise 7.6: Value Iteration on Three-tile, Straight-line Hexworld"""
    P = ThreeTileStraightLineHexWorld

    M = ValueIteration(k_max=1)
    policy_after_one_step = M.solve(P)

    print("U(s_1) =", policy_after_one_step.U[0])
    print("U(s_2) =", policy_after_one_step.U[1])
    print("U(s_3) =", policy_after_one_step.U[2], "\n")

    M.k_max = 2
    policy_after_two_steps = M.solve(P)

    print("U(s_1) =", policy_after_two_steps.U[0])
    print("U(s_2) =", policy_after_two_steps.U[1])
    print("U(s_3) =", policy_after_two_steps.U[2])

exercise_7_6()
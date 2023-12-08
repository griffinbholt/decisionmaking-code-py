import sys; sys.path.append('../');

from ch24 import NashEquilibrium, CorrelatedEquilibrium
from problems.SimpleGames import PrisonersDilemma, RockPaperScissors

def example_24_3():
    """Example 24.3: Deterministic and stochastic Nash equilibria"""
    M = NashEquilibrium()

    print("Prisoner's Dilemma: Nash Equilibrium")
    policy = M.solve(PrisonersDilemma())
    for i, policy_i in enumerate(policy):
        print("Agent " + str(i) + ": ", policy_i.p)
    print()

    print("Rock-Paper-Scissors: Nash Equilibrium")
    policy = M.solve(RockPaperScissors())
    for i, policy_i in enumerate(policy):
        print("Agent " + str(i) + ": ", policy_i.p)

def example_24_4():
    """Example 24.4: Computing correlated equilibria in rock-paper-scissors"""
    M = CorrelatedEquilibrium()

    print("Rock-Paper-Scissors: Correlated Equilibrium")
    policy = M.solve(RockPaperScissors())
    print(policy.p)

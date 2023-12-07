import sys; sys.path.append('../'); sys.path.append('../examples/')

import numpy as np


def exercise_6_3():
    """Exercise 6.3: Utility of Lotteries"""
    S = ['A', 'B', 'C']                      # Outcomes
    U_S = np.array([5, 20, 0])               # Utilities of each outcome
    lotteries = np.array([[0.0, 0.5, 0.5],   # Lottery 1
                          [1.0, 0.0, 0.0]])  # Lottery 2
    m, b = 2, 30  # Affine transformation

    U_lotteries = lotteries @ U_S
    print("Original Lottery Utilities: ", U_lotteries)
    print("Preferred Lottery: ", np.argmax(U_lotteries) + 1, "\n")

    new_U_S = m * U_S + 30
    new_U_lotteries = lotteries @ new_U_S
    print("New Lottery Utilities: ", new_U_lotteries)
    print("Preferred Lottery: ", np.argmax(new_U_lotteries) + 1)    


def exercise_6_5():
    """Exercise 6.5: Maximum Expected Utility for Simple Weather Forecast Problem"""
    from problems.SimpleWeatherForecast import P, U  # from Example 6.3

    o = 'forecast sun'  # Assumption

    exp_utility_bring = np.sum([P[(o, 'bring umbrella', s)] * U[s] for s in ['rain with umbrella', 'sun with umbrella']])
    print("Expected Utility of Bringing the Umbrella: ", exp_utility_bring)

    exp_utility_leave = np.sum([P[(o, 'leave umbrella', s)] * U[s] for s in ['rain without umbrella', 'sun without umbrella']])
    print("Expected Utility of Leaving the Umbrella: ", exp_utility_leave)

    print("Maximal Utility Action: ", 'bring umbrella' if exp_utility_bring > exp_utility_leave else 'leave_umbrella')


def exercise_6_6(print_results=False):
    """Exercise 6.6: Maximum Expected Utility for Simple Feeding Puppy Problem"""
    from problems.SimpleFeedingPuppy import U, H

    P_h_given_whining = {'not hungry': 0.22, 'hungry': 0.78}

    exp_utility_not_feed = np.sum([P_h_given_whining[h] * U[('not feed', h)] for h in H])
    exp_utility_feed = np.sum([P_h_given_whining[h] * U[('feed', h)] for h in H])

    if print_results:
        print("Expected Utility of Not Feeding: ", exp_utility_not_feed)
        print("Expected Utility of Feeding: ", exp_utility_feed)
        print("Maximal Utility Action: ", 'feed' if exp_utility_feed > exp_utility_not_feed else 'not feed')

    return max(exp_utility_feed, exp_utility_not_feed)


def exercise_6_7():
    """Exercise 6.7: VOI for Simple Feeding Puppy Problem"""
    from problems.SimpleFeedingPuppy import U, F, H, R  # from Exercise 6.6

    P_h_given_whining_r = {('not hungry', 'not fed recently'): 0.1,
                           ('hungry', 'not fed recently'):     0.9,
                           ('not hungry', 'fed recently'):     0.7,
                           ('hungry', 'fed recently'):         0.3}
    P_r_given_whining = {'not fed recently': 0.8, 'fed recently': 0.2}
    
    exp_utility_opt_action = exercise_6_6()
    opt_exp_utilities = {r: np.max([np.sum([P_h_given_whining_r[(h, r)] * U[(f, h)] for h in H]) for f in F]) for r in R}
    voi = np.sum([P_r_given_whining[r] * opt_exp_utilities[r] for r in R]) - exp_utility_opt_action

    print("VOI: ", voi)

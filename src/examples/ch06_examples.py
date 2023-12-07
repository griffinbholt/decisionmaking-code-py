import sys; sys.path.append('./src/'); sys.path.append('../')

import numpy as np


def example_6_3():
    """
    Example 6.3: Applying the principle of maximum expected utility to the
    simple decision of whether to bring an umbrella
    """
    from problems.WeatherForecastSimpleProblem import P, U

    o = 'forecast rain'  # Assumption

    exp_utility_bring = np.sum([P[(o, 'bring umbrella', s)] * U[s] for s in ['rain with umbrella', 'sun with umbrella']])
    print("Expected Utility of Bringing the Umbrella: ", exp_utility_bring)

    exp_utility_leave = np.sum([P[(o, 'leave umbrella', s)] * U[s] for s in ['rain without umbrella', 'sun without umbrella']])
    print("Expected Utility of Leaving the Umbrella: ", exp_utility_leave)

    print("Maximal Utility Action: ", 'bring umbrella' if exp_utility_bring > exp_utility_leave else 'leave_umbrella')

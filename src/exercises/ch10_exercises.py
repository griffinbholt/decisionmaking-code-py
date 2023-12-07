import sys; sys.path.append('../')

from ch10 import evolution_strategy_weights

def exercise_10_8():
    """Exercise 10.8: Evolution Strategy Weights"""
    w = evolution_strategy_weights(m=3)
    print(w)

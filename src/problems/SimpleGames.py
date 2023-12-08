import sys; sys.path.append('./src/'); sys.path.append('../'); sys.path.append('../../')

import numpy as np

from ch24 import SimpleGame


class PrisonersDilemma(SimpleGame):
    def __init__(self):
        gamma = 0.9
        I = [0, 1]
        A = [['cooperate', 'defect'], ['cooperate', 'defect']]
        def R(a):
            if a[0] == 'cooperate' and a[1] == 'cooperate':
                return np.array([-1, -1])
            elif a[0] == 'cooperate' and a[1] == 'defect':
                return np.array([-4, 0])
            elif a[0] == 'defect' and a[1] == 'cooperate':
                return np.array([0, -4])
            else: # both 'defect'
                return np.array([-3, -3])
        super().__init__(gamma, I, A, R)


class RockPaperScissors(SimpleGame):
    def __init__(self):
        gamma = 0.9
        I = [0, 1]
        A = [['rock', 'paper', 'scissors'], ['rock', 'paper', 'scissors']]
        def R(a):
            if a[0] == a[1]:
                return np.zeros(2)
            elif a[0] == 'rock' and a[1] == 'paper':
                return np.array([-1, 1])
            elif a[0] == 'rock' and a[1] == 'scissors':
                return np.array([1, -1])
            elif a[0] == 'paper' and a[1] == 'rock':
                return np.array([1, -1])
            elif a[0] == 'paper' and a[1] == 'scissors':
                return np.array([-1, 1])
            elif a[0] == 'scissors' and a[1] == 'rock':
                return np.array([-1, 1])
            else:  # 'scissors', 'paper
                return np.array([1, -1])
        super().__init__(gamma, I, A, R)

class AlmostRockPaperScissors(SimpleGame):
    def __init__(self):
        gamma = 0.9
        I = [0, 1]
        A = [['rock', 'paper', 'scissors'], ['rock', 'paper', 'scissors']]
        def R(a):
            if a[0] == a[1]:
                return np.zeros(2)
            elif a[0] == 'rock' and a[1] == 'paper':
                return np.array([0, 1])
            elif a[0] == 'rock' and a[1] == 'scissors':
                return np.array([1, 0])
            elif a[0] == 'paper' and a[1] == 'rock':
                return np.array([1, 0])
            elif a[0] == 'paper' and a[1] == 'scissors':
                return np.array([0, 1])
            elif a[0] == 'scissors' and a[1] == 'rock':
                return np.array([0, 1])
            else:  # 'scissors', 'paper
                return np.array([1, 0])
        super().__init__(gamma, I, A, R)


class TravelersDilemma(SimpleGame):
    def __init__(self):
        gamma = 0.9
        I = [0, 1]  # two agents
        A = [[i for i in range(2, 101, 2)], [i for i in range(2, 101, 2)]]  # $2 - $100
        def R(a):
            if a[0] == a[1]:
                return np.array([a[0], a[1]])
            elif a[0] < a[1]:
                return np.array([a[0] + 2, a[0] - 2])
            else:  # a[0] > a[1]
                return np.array([a[1] - 2, a[1] + 2])
        super().__init__(gamma, I, A, R)

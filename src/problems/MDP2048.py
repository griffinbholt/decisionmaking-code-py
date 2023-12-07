import sys; sys.path.append('../');

import numpy as np
import random

from ch07 import MDP

class MDP2048(MDP):
    N = 4
    COORDS = [(i, j) for i in range(4) for j in range(4)]
    ACTIONS = ["left", "down", "right", "up"]
    BASE_VALS = [2, 4]

    def __init__(self, gamma: float):
        ordered_actions = [i for i in range(len(MDP2048.ACTIONS))]
        pass

def generate_init_state() -> np.ndarray:
    # State is a 4x4 matrix
    s = np.zeros((MDP2048.N, MDP2048.N))
    selected = random.sample(MDP2048.COORDS, 2)
    for idx in selected:
        s[idx] = random.choices(MDP2048.BASE_VALS)[0]
    return s

def transition(s: np.ndarray, a: int) -> np.ndarray:
    # TODO - write the code for the swipe behavior
    
    # Generate new random tile on empty space with value 2 or 4
    zero_idcs_x, zero_idcs_y = np.where(s == 0)
    selected = random.choices(list(zip(zero_idcs_x, zero_idcs_y)))[0]
    s[selected] = random.choices(MDP2048.BASE_VALS)[0]

    return s
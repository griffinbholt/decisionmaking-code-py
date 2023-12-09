import sys; sys.path.append('../');

import numpy as np
import random

from ch07 import MDP

class MDP2048(MDP):
    N = 4
    COORDS = [(i, j) for i in range(4) for j in range(4)]
    ACTIONS = ["left", "down", "right", "up"]
    BASE_VALS = [2, 4]

    def generate_start_state() -> np.ndarray:
        # State is a 4x4 matrix
        s = np.zeros((MDP2048.N, MDP2048.N))
        selected = random.sample(MDP2048.COORDS, 2)
        for idx in selected:
            s[idx] = random.choices(MDP2048.BASE_VALS)[0]
        return s

    def transition(s: np.ndarray, a: int) -> np.ndarray:
        def move_right(s):
            s_prime, changed_1 = compress(s)
            s_prime, changed_2 = merge(s_prime)
            changed = changed_1 or changed_2
            s_prime, _ = compress(s_prime)
            return s_prime, changed

        def move_left(s):
            s_prime = np.fliplr(s)
            s_prime, changed = move_right(s_prime)
            s_prime = np.fliplr(s_prime)
            return s_prime, changed

        def move_down(s):
            s_prime = s.T
            s_prime, changed = move_right(s_prime)
            s_prime = s_prime.T
            return s_prime, changed

        def move_up(s):
            s_prime = s.T
            s_prime, changed = move_left(s_prime)
            s_prime = s_prime.T
            return s_prime, changed

        def compress(s):
            s_prime = np.take_along_axis(s, np.argsort(s.astype(bool), axis=1), axis=1)
            changed = np.any(s != s_prime)
            return s_prime, changed

        def merge(s):
            changed = False
            for i in range(MDP2048.N):
                for j in range(MDP2048.N - 1, 0, -1):
                    if (s[i, j] == s[i, j - 1]) and (s[i, j] != 0):
                        s[i, j] *= 2
                        s[i, j - 1] = 0
                        changed = True
            return s, changed

        if a == 0:
            s_prime, changed = move_left(s)
        elif a == 1:
            s_prime, changed = move_down(s)
        elif a == 2:
            s_prime, changed = move_right(s)
        else:
            s_prime, changed = move_up(s)
        
        if changed:
            # Generate new random tile on empty space with value 2 or 4
            zero_idcs_x, zero_idcs_y = np.where(s_prime == 0)
            selected = random.choices(list(zip(zero_idcs_x, zero_idcs_y)))[0]
            s_prime[selected] = random.choices(MDP2048.BASE_VALS)[0]

        return s_prime

    def __init__(self, gamma: float):
        ordered_actions = [i for i in range(len(MDP2048.ACTIONS))]
        pass  # TODO

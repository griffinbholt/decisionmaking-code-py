import sys; sys.path.append('./src/'); sys.path.append('../'); sys.path.append('../../')

import numpy as np

from typing import Dict, List, Tuple

from ch07 import MDP


class HexWorldMDP(MDP):
    def __init__(self, 
                 hexes: List[Tuple[int, int]],                       # The coordinates of the states (hexes)
                 r_bump_border: float,                               # Reward for bumping up against the border
                 p_intended: float,                                  # Probability of going where intended (same for all states)
                 special_hex_rewards: Dict[Tuple[int, int], float],  # Dictionary with {Reward State : Reward}
                 gamma: float                                        # Discount factor
                 ):
        n_states = len(hexes) + 1   # Hexes plus one terminal state
        n_actions = 6               # Six directions: 0 is east, 1 is north east, 2 is north west, 3 is west, 4 is south west, 5 is south east
        s_absorbing = n_states - 1  # Absorbing state

        T = np.zeros(shape=(n_states, n_actions, n_states))
        R = np.zeros(shape=(n_states, n_actions))

        p_veer = (1.0 - p_intended) / 2  # Odds of veering left or right

        for s in range(n_states - 1):
            hex = hexes[s]
            if hex not in special_hex_rewards:
                # Action taken from a normal tile
                neighbors = self.hex_neighbors(hex)
                for (a, neighbor) in enumerate(neighbors):
                    # Intended transition
                    if neighbor in hexes:
                        s_prime = hexes.index(neighbor)
                    else:  # Off the map!
                        s_prime = s
                        R[s, a] += r_bump_border * p_intended
                    T[s, a, s_prime] += p_intended

                    # Unintended veer left
                    a_left = (a + 1) % n_actions
                    neighbor_left = neighbors[a_left]
                    if neighbor_left in hexes:
                        s_prime = hexes.index(neighbor_left)
                    else:  # Off the map!
                        s_prime = s
                        R[s, a] += r_bump_border * p_veer
                    T[s, a, s_prime] += p_veer

                    # Unintended veer right
                    a_right = (a - 1) % n_actions
                    neighbor_right = neighbors[a_right]
                    if neighbor_right in hexes:
                        s_prime = hexes.index(neighbor_right)
                    else:  # Off the map!
                        s_prime = s
                        R[s, a] += r_bump_border * p_veer
                    T[s, a, s_prime] += p_veer
            else:
                # Action taken from an absorbing hex
                # In absorbing hex, your action automatically takes you to the absorbing state and you get the reward.
                for a in range(n_actions):
                    T[s, a, s_absorbing] = 1.0
                    R[s, a] += special_hex_rewards[hex]
        
        # Absorbing state stays where it is and gets no reward.
        for a in range(n_actions):
            T[s_absorbing, a, s_absorbing] = 1.0

        # Initialize the MDP
        super().__init__(gamma, [s for s in range(n_states)], [a for a in range(n_actions)], T, R)

    def hex_neighbors(self, hex: Tuple[int, int]):
        i, j = hex
        return [(i + 1, j),      # East
                (i, j + 1),      # Northeast
                (i - 1, j + 1),  # Northwest
                (i - 1, j),      # West
                (i, j - 1),      # Southwest
                (i + 1, j - 1)]  # Southeast


HexWorld = HexWorldMDP(
    hexes=[
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(-1,2),
        (0,2),(1,2),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2),
        (8,2),(4,1),(5,0),(6,0),(7,0),(7,1),(8,1),(9,0)
    ],
    r_bump_border=-1.0,  # Reward for falling off hex map
    p_intended=0.7,      # Probability of going in intended direction (same for each state)
    special_hex_rewards={
        (0,1): 5.0,    # left side reward
        (2,0): -10.0,  # left side hazard
        (9,0): 10.0    # right side reward
    },
    gamma=0.9            # Discount factor
)

StraightLineHexWorld = HexWorldMDP(
    hexes=[(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0)],
    r_bump_border=-1.0,
    p_intended=0.7,
    special_hex_rewards={
        (6,0): 10.0  # right side reward
    },
    gamma=0.9
)


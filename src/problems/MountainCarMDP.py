import sys; sys.path.append('./src/'); sys.path.append('../'); sys.path.append('../../')

import numpy as np

from scipy.stats import multivariate_normal

from ch07 import MDP


class MountainCarMDP(MDP):
    # State is a 2-vector of position and velocity.
    STATE_MINS = np.array([-1.2, -0.07])
    STATE_MAXS = np.array([ 0.6,  0.07])
    ACCELS = np.array([-1.0, 0.0, 1.0])
    START_STATE = np.array([-0.5, 0.0])

    def generate_start_state():
        x = MountainCarMDP.STATE_MINS[0] + np.random.rand() * (MountainCarMDP.STATE_MAXS[0] - MountainCarMDP.STATE_MINS[0])
        v = MountainCarMDP.STATE_MINS[1] + np.random.rand() * (MountainCarMDP.STATE_MAXS[1] - MountainCarMDP.STATE_MINS[1])
        return np.array([x, v])

    def __init__(self, gamma: float):
        ordered_actions = [a for a in range(len(MountainCarMDP.ACCELS))]

        # NOTE: Mountain car is deterministic, so we don't get a transition
        # function that produces a probability distribution
        def transition(s: np.ndarray, a: int) -> np.ndarray:
            x, v = s
            if x < MountainCarMDP.STATE_MAXS[0]:
                accel = MountainCarMDP.ACCELS[a]
                v += 0.001*accel - 0.0025*np.cos(3*x)
                x += v
                x = np.clip(x, MountainCarMDP.STATE_MINS[0], MountainCarMDP.STATE_MAXS[0])
                v = np.clip(v, MountainCarMDP.STATE_MINS[1], MountainCarMDP.STATE_MAXS[1])
            return np.array([x, v])
    
        def T(s, a, s_prime=None):
            S_prime = multivariate_normal(transition(s, a), 1e-6)  # Technically wrong, but its probably fine
            if s_prime is None:
                return S_prime
            return S_prime.pdf(s_prime)

        def R(s: np.ndarray, a: int) -> float:
            if s[0] >= MountainCarMDP.STATE_MAXS[0]:
                return 0.0
            return -1.0

        def TR(s: np.ndarray, a: int) -> tuple[np.ndarray, float]:
            s_prime = transition(s, a)
            r = R(s, a)
            return (s_prime, r)
        
        super().__init__(gamma, None, ordered_actions, T, R, TR)


def mountain_car_expert_policy(s: np.ndarray) -> int:
    v = s[1]  # velocity
    if v < 0.0:
        return 0  # accel left
    elif v > 0.0:
        return 2  # accel right
    return 1  # coast


def eval_mtn_car_expert_policy(s_init: np.ndarray, max_iter: int = 2000) -> float:
    s = s_init.copy()
    if s[0] >= MountainCarMDP.STATE_MAXS[0]:
        return 0.0
    
    r = 0.0
    for _ in range(max_iter):
        a = mountain_car_expert_policy(s)

        x, v = s
        accel = MountainCarMDP.ACCELS[a]
        v += 0.001*accel - 0.0025*np.cos(3*x)
        x += v
        if x >= MountainCarMDP.STATE_MAXS[0]:
            return r

        x = np.clip(x, MountainCarMDP.STATE_MINS[0], MountainCarMDP.STATE_MAXS[0])
        v = np.clip(v, MountainCarMDP.STATE_MINS[1], MountainCarMDP.STATE_MAXS[1])
        s = np.array([x, v])
        r -= 1.0
    return r

MountainCar = MountainCarMDP(gamma=0.9999)
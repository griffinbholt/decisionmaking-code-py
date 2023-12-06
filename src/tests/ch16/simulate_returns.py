import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np

from ch07 import MDP
from ch16 import ModelBasedMDP, RLPolicy

def simulate_returns(P: MDP, model: ModelBasedMDP, policy: RLPolicy, h: int, n: int):
    returns = np.zeros(n)
    for t in range(n):
        rr = 0.0
        s = P.S[0]
        for _ in range(h):
            a = policy(model, s)
            s_prime, r = P.TR(s, a)
            rr += r
            model.update(s, a, r, s_prime)
            s = s_prime
        returns[t] = rr
    return returns
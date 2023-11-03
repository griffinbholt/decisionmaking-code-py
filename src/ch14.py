import numpy as np

from typing import Any, Callable

from ch07 import MDP


def adversarial(P: MDP, policy: Callable[[Any], Any], lam: float) -> MDP:
    S, T, R, gamma = P.S, P.T, P.R, P.gamma
    S_prime = A_prime = S
    R_prime = np.zeros((len(S_prime), len(A_prime)))
    T_prime = np.zeros((len(S_prime), len(A_prime), len(S_prime)))
    for s in S_prime:
        for a in A_prime:
            R_prime[s, a] = -R(s, policy(s)) + (lam * np.log(T(s, policy(s), a)))
            T_prime[s, a, a] = 1.0
    return MDP(gamma, S_prime, A_prime, T_prime, R_prime)

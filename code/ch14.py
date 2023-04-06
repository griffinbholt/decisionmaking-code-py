import numpy as np

from typing import Any, Callable

from ch07 import MDP


def adversarial(P: MDP, policy: Callable[[Any], Any], lam: float) -> MDP:
    S, A, T, R, gamma = P.S, P.A, P.T, P.R, P.gamma
    S_prime = A_prime = S
    R_prime_matrix = np.zeros((len(S_prime), len(A_prime)))
    T_prime_matrix = np.zeros((len(S_prime), len(A_prime), len(S_prime)))
    for s in S_prime:
        for a in A_prime:
            R_prime_matrix[s, a] = -R(s, policy(s)) + (lam * np.log(T(s, policy(s), a)))
            T_prime_matrix[s, a, a] = 1.0
    T_prime = lambda s, a, s_prime: T_prime_matrix[s, a, s_prime] # TODO - Consider making a function inside of MDP constructor to convert matrices to functions
    R_prime = lambda s, a: R_prime_matrix[s, a]
    TR_prime = lambda s, a: (np.random.choice(len(S_prime), p=T_prime_matrix[s, a]), R_prime(s, a))
    return MDP(gamma, S_prime, A_prime, T_prime, R_prime, TR_prime)

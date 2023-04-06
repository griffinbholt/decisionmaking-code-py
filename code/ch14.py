import numpy as np

from typing import Any, Callable

from ch07 import MDP


def adversarial(P: MDP, policy: Callable[[Any], Any], lam: float) -> MDP:
    S, T, R, gamma = P.S, P.T, P.R, P.gamma
    S_prime = A_prime = S
    R_prime_matrix = np.zeros((len(S_prime), len(A_prime)))
    T_prime_matrix = np.zeros((len(S_prime), len(A_prime), len(S_prime)))
    for s in S_prime:
        for a in A_prime:
            R_prime_matrix[s, a] = -R(s, policy(s)) + (lam * np.log(T(s, policy(s), a)))
            T_prime_matrix[s, a, a] = 1.0
    # TODO - Consider making a function inside of MDP constructor to convert matrices to functions
    def T_prime(s, a, s_prime): return T_prime_matrix[s, a, s_prime]
    def R_prime(s, a): return R_prime_matrix[s, a]
    def TR_prime(s, a): return (np.random.choice(len(S_prime), p=T_prime_matrix[s, a]), R_prime(s, a))
    return MDP(gamma, S_prime, A_prime, T_prime, R_prime, TR_prime)

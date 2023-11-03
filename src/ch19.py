import jax.numpy as jnp
import numpy as np
import random

from jax import jacfwd
from typing import Any, Callable

from convenience import normalize


class POMDP():
    def __init__(self,
                 gamma: float,
                 S: list[Any],
                 A: list[Any],
                 O_space: list[Any],
                 T: Callable[[Any, Any, Any], float],
                 R: Callable[[Any, Any], float],
                 O: Callable[[Any, Any, Any], float],
                 TRO: Callable[[Any, Any], tuple[Any, float, Any]]):
        self.gamma = gamma      # discount factor
        self.S = S              # state space
        self.A = A              # action space
        self.O_space = O_space  # observation space
        self.T = T              # transition function
        self.R = R              # reward function
        self.O = O              # observation function
        self.TRO = TRO          # sample next state, reward, and observation given current state and action: s', r, o = TRO(s, a)

    def lookahead_from_state(self, U: Callable[[Any, Any], float], s: Any, a: Any) -> float:
        S, O_space, T, O, R, gamma = self.S, self.O_space, self.T, self.O, self.R, self.gamma
        u = np.sum([T(s, a, s_prime) * np.sum([O(a, s_prime, o) * U(o, s_prime) for o in O_space]) for s_prime in S])
        return R(s, a) + gamma * u

    def lookahead_from_belief(self, U: Callable[[Any, Any], float], b: np.ndarray, a: Any) -> float:
        S, O_space, T, O, R, gamma = self.S, self.O_space, self.T, self.O, self.R, self.gamma
        dsf = DiscreteStateFilter(b)
        r = np.sum([R(s, a) * b[i] for (i, s) in enumerate(S)])
        def P_osa(o, s, a): return np.sum([O(a, s_prime, o) * T(s, a, s_prime) for s_prime in S])
        def P_oba(o, b, a): return np.sum([b[i] * P_osa(o, s, a) for (i, s) in enumerate(S)])
        return r + gamma * np.sum([P_oba(o, b, a) * U(dsf.update(self, a, o).b) for o in O_space])

    def greedy(self, U: Callable[[Any, Any], float], b: np.ndarray):
        expected_rewards = [self.lookahead_from_belief(U, b, a) for a in self.A]
        idx = np.argmax(expected_rewards)
        return self.A[idx], expected_rewards[idx]

    def combine_lookahead(self, s: Any, a: Any, Vo: np.ndarray) -> float:
        S, O_space, T, O, R, gamma = self.S, self.O_space, self.T, self.O, self.R, self.gamma
        def U_prime(s_prime, i): return np.sum([O(a, s_prime, o) * alpha[i] for (o, alpha) in zip(O_space, Vo)])
        return R(s, a) + gamma * np.sum([T(s, a, s_prime) * U_prime(s_prime, i) for (i, s_prime) in enumerate(S)])

    def combine_alphavector(self, a: Any, Vo: np.ndarray) -> np.ndarray:
        return np.ndarray([self.combine_lookahead(s, a, Vo) for s in self.S])

    def backup(self, V: np.ndarray, b: np.ndarray) -> np.ndarray:
        S, A, O_space, gamma = self.S, self.A, self.O_space, self.gamma
        R, T, O = self.R, self.T, self.O
        dsf = DiscreteStateFilter(b)
        Va = []
        for a in A:
            Vao = []
            for o in O_space:
                b_prime = dsf.update(self, a, o).b
                idx = np.argmax([np.dot(alpha, b_prime) for alpha in V])
                Vao.append(V[idx])
            alpha = np.array([R(s, a) + gamma * np.sum([np.sum([T(s, a, s_prime) * O(a, s_prime, o) * Vao[i][j] for (j, s_prime) in enumerate(S)]) for (i, o) in enumerate(O_space)]) for s in S])
            Va.append(alpha)
        idx = np.argmax([np.dot(alpha, b) for alpha in Va])
        return Va[idx]

    def randstep(self, b: np.ndarray, a: Any) -> tuple[np.ndarray, float]:
        dsf = DiscreteStateFilter(b)
        s = random.choices(self.S, weights=b)
        s_prime, r, o = self.TRO(s, a)
        b_prime = dsf.update(self, a, o).b
        return b_prime, r


class DiscreteStateFilter():
    def __init__(self, b: np.ndarray):
        self.b = b  # discrete state belief

    def update(self, P: POMDP, a: Any, o: Any) -> 'DiscreteStateFilter':
        b_prime = np.zeros_like(self.b)
        for (i_prime, s_prime) in enumerate(P.S):
            prob_o = P.O(a, s_prime, o)
            b_prime = prob_o * np.sum([P.T(s, a, s_prime) * self.b[i] for (i, s) in enumerate(self.S)])
        if np.isclose(np.sum(b_prime), 0.0):
            b_prime = np.ones_like(self.b)
        return DiscreteStateFilter(normalize(b_prime, ord=1))


class KalmanFilter():
    def __init__(self, mu_b: np.ndarray, Sigma_b: np.ndarray):
        self.mu_b = mu_b        # mean vector
        self.Sigma_b = Sigma_b  # covariance matrix

    def update(self, P: POMDP, a: np.ndarray, o: np.ndarray) -> 'KalmanFilter':
        Ts, Ta, Os = P.Ts, P.Ta, P.Os
        Sigma_s, Sigma_o = P.Sigma_s, P.Sigma_o

        # Predict
        mu_p = (Ts @ self.mu_b) + (Ta @ a)
        Sigma_p = (Ts @ self.Sigma_b @ Ts.T) + Sigma_s

        # Update
        Sigma_po = Sigma_p @ Os.T
        K = Sigma_po @ np.linalg.inv((Os @ Sigma_p @ Os.T) + Sigma_o)
        mu_b_prime = mu_p + K @ (o - (Os @ mu_p))
        Sigma_b_prime = (np.eye(K.shape[0]) - (K @ Os)) @ Sigma_p

        return KalmanFilter(mu_b_prime, Sigma_b_prime)


class ExtendedKalmanFilter(KalmanFilter):  # TODO - Test use of Jax
    def __init__(self, mu_b: np.ndarray, Sigma_b: np.ndarray):
        super().__init__(mu_b, Sigma_b)

    def update(self, P: POMDP, a: np.ndarray, o: np.ndarray) -> 'ExtendedKalmanFilter':
        fT, fO = P.fT, P.fO  # fT, fO need to be jax compatible (i.e., they have to handle `jax.numpy` arrays)
        Sigma_s, Sigma_o = P.Sigma_s, P.Sigma_o

        # Predict
        mu_p = fT(self.mu_b, a)
        Ts = np.array(jacfwd(lambda s: fT(s, a))(jnp.asarray(self.mu_b)))
        Os = np.array(jacfwd(fO)(jnp.asarray(mu_p)))
        Sigma_p = (Ts @ self.Sigma_b @ Ts.T) + Sigma_s

        # Update
        Sigma_po = Sigma_p @ Os.T
        K = Sigma_po @ np.linalg.inv((Os @ Sigma_p @ Os.T) + Sigma_o)
        mu_b_prime = mu_p + K @ (o - fO(mu_p))
        I = np.eye(K.shape[0])
        Sigma_b_prime = (I - (K @ Os)) @ Sigma_p

        return ExtendedKalmanFilter(mu_b_prime, Sigma_b_prime)


class UnscentedKalmanFilter(KalmanFilter):
    def __init__(self, mu_b: np.ndarray, Sigma_b: np.ndarray, lam: float):
        super().__init__(mu_b, Sigma_b)
        self.lam = lam  # spread parameter

    def update(self, P: POMDP, a: np.ndarray, o: np.ndarray) -> 'UnscentedKalmanFilter':
        pass  # TODO

    def unscented_transform(self, mu, Sigma, f, ws):  # TODO - Typing
        pass  # TODO


class ParticleFilter():
    def __init__(self, states):
        self.states = states  # vector of state samples

    def update(self, P: POMDP, a: np.ndarray, o: np.ndarray) -> 'ParticleFilter':
        pass  # TODO


class RejectionParticleFilter(ParticleFilter):
    def __init__(self, states):
        super().__init__(states)

    def update(self, P: POMDP, a: np.ndarray, o: np.ndarray) -> 'RejectionParticleFilter':
        pass  # TODO


class InjectionParticleFilter(ParticleFilter):
    def __init__(self, states, m_inject: int, D_inject):
        super().__init__(states)
        self.m_inject = m_inject  # number of samples to inject
        self.D_inject = D_inject  # injection distribution

    def update(self, P: POMDP, a: np.ndarray, o: np.ndarray) -> 'InjectionParticleFilter':
        pass  # TODO


class AdaptiveInjectionParticleFilter(ParticleFilter):
    def __init__(self, states, w_slow, w_fast, a_slow, a_fast, v, D_inject):
        super().__init__(states)
        self.w_slow = w_slow      # slow moving average
        self.w_fast = w_fast      # fast moving average
        self.a_slow = a_slow      # slow moving average parameter
        self.a_fast = a_fast      # fast moving average parameter
        self.v = v                # injection parameter
        self.D_inject = D_inject  # injection distribution

    def update(self, P: POMDP, a: np.ndarray, o: np.ndarray) -> 'AdaptiveInjectionParticleFilter':
        pass  # TODO

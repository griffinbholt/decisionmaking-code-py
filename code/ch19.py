
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
        self.gamma = gamma # discount factor
        self.S = S # state space
        self.A = A # action space
        self.O_space = O_space # observation space
        self.T = T # transition function
        self.R = R # reward function
        self.O = O # observation function
        self.TRO = TRO # sample next state, reward, and observation given current state and action: s', r, o = TRO(s, a)

class DiscreteStateFilter():
    def __init__(self, b: np.ndarray):
        self.b = b # discrete state belief

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
        self.mu_b = mu_b # mean belief
        self.Sigma_b = Sigma_b # covariance belief

    def update(self, P: POMDP, a: np.ndarray, o: np.ndarray) -> 'KalmanFilter':
        mu_p, Sigma_p = self._predict_step(P, a)
        mu_b_prime, Sigma_b_prime = self._update_step(P, o, mu_p, Sigma_p)
        return KalmanFilter(mu_b_prime, Sigma_b_prime)

    def _predict_step(self, P: POMDP, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Ts, Ta, Sigma_s = P.Ts, P.Ta, P.Sigma_s
        mu_p = (Ts @ self.mu_b) + (Ta @ a)
        Sigma_p = (Ts @ self.Sigma_b @ Ts.T) + Sigma_s
        return mu_p, Sigma_p

    def _update_step(self, P: POMDP, o: np.ndarray, mu_p: np.ndarray, Sigma_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Os, Sigma_o = P.Os, P.Sigma_o         
        Sigma_po = Sigma_p @ Os.T
        K = np.linalg.solve(Sigma_po, ((Os @ Sigma_p @ Os.T) + Sigma_o))
        mu_b_prime = mu_p + K @ (o - (Os @ mu_p))
        Sigma_b_prime = (np.eye(K.shape[0]) - (K @ Os)) @ Sigma_p
        return mu_b_prime, Sigma_b_prime

class ExtendedKalmanFilter(KalmanFilter):
    def __init__(self, mu_b: np.ndarray, Sigma_b: np.ndarray):
        super().__init__(mu_n, Sigma_b)

    # TODO - If I just overwrite the predict_step and update_step methods, and then
    # use the super in update, wrapped by Extended Kalman Filter, will it work?
    def update(self, P: POMDP, a: np.ndarray, o: np.ndarray) -> 'ExtendedKalmanFilter':
        mu_p, Sigma_p = self._predict_step(P, a)
        mu_b_prime, Sigma_b_prime = self._update_step(P, o, mu_p, Sigma_p)
        return ExtendedKalmanFilter(mu_b_prime, Sigma_b_prime)

    def _predict_step(self, P: POMDP, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass # TODO

    def _update_step(self, P: POMDP, o: np.ndarray, mu_p: np.ndarray, Sigma_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass # TODO
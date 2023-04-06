import numpy as np 
import random

from collections import deque
from typing import Callable

from ch12 import scale_gradient
from ch16 import RLMDP

class IncrementalEstimate():
    def __init__(self, mu: float | np.ndarray, alpha: Callable[[int], float], m: int):
        self.mu = mu       # mean estimate
        self.alpha = alpha # learning rate function
        self.m = m         # number of updates
    
    def update(self, x: float | np.ndarray):
        self.m += 1
        self.mu += self.alpha(self.m) * (x - self.mu)

class ModelFreeMDP(RLMDP):
    def __init__(self, A: list[int], gamma: float, Q: np.ndarray | Callable[[np.ndarray, float, int], float], alpha: float):
        super().__init__(A, gamma)
        self.Q = Q         # action value function, either as a numpy array Q[s, a] or a parametrized function Q(theta, s, a) (depending on method)
        self.alpha = alpha # learning rate

class QLearning(ModelFreeMDP):
    def __init__(self, S: list[int], A: list[int], gamma: float, Q: np.ndarray, alpha: float):
        super().__init__(A, gamma, Q, alpha) # The action value function Q[s, a] is a numpy array
        self.S = S # state space (assumes 1:nstates)
        
    def lookahead(self, s: int, a: int):
        return self.Q[s, a]

    def update(self, s: int, a: int, r: float, s_prime: int):
        self.Q[s, a] += self.alpha * (r + (self.gamma * np.max(self.Q[s_prime])) - self.Q[s, a])
    
class Sarsa(ModelFreeMDP):
    def __init__(self, S: list[int], A: list[int], gamma: float, Q: np.ndarray, alpha: float, ell: tuple[int, int, float]):
        super().__init__(A, gamma, Q, alpha) # The action value function Q[s, a] is a numpy array
        self.S = S     # state space (assumes 1:nstates)
        self.ell = ell # most recent experience tuple (s, a, r)

    def lookahead(self, s: int, a: int):
        return self.Q[s, a]

    def update(self, s: int, a: int, r: float, s_prime: int):
        if self.ell is not None:
            s_prev, a_prev, r_prev = self.ell
            self.Q[s_prev, a_prev] += self.alpha * (r_prev + (self.gamma * self.Q[s, a]) - self.Q[s_prev, a_prev])
        self.ell = (s, a, r)

class SarsaLambda(Sarsa):
    def __init__(self, S: list[int], A: list[int], gamma: float, Q: np.ndarray, N: np.ndarray, alpha: float, lam: float, ell: tuple[int, int, float]):
        super().__init__(S, A, gamma, Q, alpha, ell) # The action value function Q[s, a] is a numpy array
        self.N = N     # trace
        self.lam = lam # trace decay rate

    # TODO - Tell that there is overloading in the Julia code of variable names
    def update(self, s: int, a: int, r: float, s_prime: int):
        if self.ell is not None:
            s_prev, a_prev, r_prev = self.ell
            self.N[s_prev, a_prev] += 1
            delta = r_prev + (self.gamma * self.Q[s, a]) - self.Q[s_prev, a_prev]
            self.Q += (self.alpha * delta * self.N)
            self.N *= (self.gamma * self.lam)
        else:
            self.N = np.zeros_like(self.Q)
        self.ell = (s, a, r)

class GradientQLearning(ModelFreeMDP):
    def __init__(self, A: list[int], gamma: float, Q: Callable[[np.ndarray, float, int], float], grad_Q: Callable[[np.ndarray, float, int], float], theta: np.ndarray, alpha: float):
        super().__init__(A, gamma, Q, alpha) # The action value function is parametrized Q(theta, s, a)
        self.grad_Q = grad_Q
        self.theta = theta
    
    # Note that s can be a float, for continuous state spaces
    def lookahead(self, s: float, a: int):
        return self.Q(self.theta, s, a)

    # Note that s, s_prime can be floats, for continuous state spaces
    def update(self, s: float, a: int, r: float, s_prime: float):
        u = np.max([self.Q(self.theta, s_prime, a_prime) for a_prime in self.A])
        delta = (r + self.gamma*u - self.Q(self.theta, s, a)) * self.grad_Q(self.theta, s, a)
        self.theta += self.alpha * scale_gradient(delta, l2_max=1.0)

class ReplayGradientQLearning(GradientQLearning):
    def __init__(self, A: list[int], gamma: float, Q: Callable[[np.ndarray, float, int], float], grad_Q: Callable[[np.ndarray, float, int], float], theta: np.ndarray, alpha: float, buffer: deque, m: int, m_grad: int):
        super().__init__(A, gamma, Q, grad_Q, theta, alpha) # The action value function is parametrized Q(theta, s, a)
        self.buffer = buffer # circular memory buffer, in the form of a collections.deque object with maxlen capacity
        self.m = m # number of steps between gradient updates
        self.m_grad = m_grad # batch size

    # Note that s, s_prime can be floats, for continuous state spaces
    def update(self, s: float, a: int, r: float, s_prime: float):
        if len(self.buffer) == self.buffer.maxlen: # i.e., the buffer is full
            U = lambda s: np.max([self.Q(self.theta, s, a) for a in self.A])
            del_Q = lambda s, a, r, s_prime: (r + (self.gamma * U(s_prime)) - self.Q(self.theta, s, a)) * self.grad_Q(self.theta, s, a)
            delta = np.mean([del_Q(s, a, r, s_prime) for (s, a, r, s_prime) in random.choices(list(self.buffer), k=self.m_grad)])
            self.theta += self.alpha * scale_gradient(delta, l2_max=1.0)
            for _ in range(self.m):
                self.buffer.popleft()
        else:
            self.buffer.append((s, a, r, s_prime))

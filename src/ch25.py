import casadi
import numpy as np
import random

from abc import ABC, abstractmethod
from typing import Any, Callable

from ch07 import MDP, ExactSolutionMethod, LinearProgramFormulation
from ch24 import joint_action, joint_policy, SimpleGamePolicy

class TODO():  # Generic to fix class types
    pass

class MG():  # TODO - Determine if it should inherit from SimpleGame or not
    """Data structure for an MG"""
    def __init__(self, gamma: float, I: list[Any], S: list[Any], A: list[list[Any]], T: TODO, R: TODO):
        self.gamma = gamma  # discount factor
        self.I = I          # agents
        self.S = S          # state space
        self.A = A          # joint action space
        self.T = T          # transition function
        self.R = R          # joint reward function

    def tensorform(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        joint_A = joint_action(self.A)
        I_tensor = np.arange(len(self.I))
        S_tensor = np.arange(len(self.S))
        A_tensor = np.array([np.arange(len(self.A[i])) for i in range(len(self.I))])
        R_tensor = np.array([[self.R(s, a) for a in joint_A] for s in self.S])
        T_tensor = np.array([[[self.T(s, a, s_prime) for s_prime in self.S] for a in joint_A] for s in self.S])
        return I_tensor, S_tensor, A_tensor, R_tensor, T_tensor

    def probability(self, s: Any, policy: TODO, a: Any) -> float:
        return np.prod([policy_j(a_j, s) for policy_j, a_j in zip(policy, a)])

    def reward(self, s: Any, policy: TODO, i: int) -> float:
        """Computes R^i(s, policy(s))"""
        return np.sum([self.R(s, a)*self.probability(s, policy, a) for a in joint_action(self.A)])

    def transition(self, s: Any, policy: TODO, s_prime: Any) -> float:
        """Computes T(s_prime | s, policy(s))"""
        return np.sum([self.T(s, a, s_prime) * self.probability(s, policy, a) for a in joint_action(self.A)])

    def policy_evaluation(self, policy: TODO, i: int) -> np.ndarray:
        """Computes a vector representing U^{policy,i}"""
        R_prime = np.array([self.reward(s, policy, i) for s in self.S])
        T_prime = np.array([[self.transition(s, policy, s_prime) for s_prime in self.S] for s in self.S])
        I = np.eye(len(self.S))
        return np.linalg.solve(I - self.gamma * T_prime, R_prime)

    def best_response(self, policy: TODO, i: int, M: ExactSolutionMethod = LinearProgramFormulation()) -> 'MGPolicy':
        """
        For an MG, we can compute a deterministic best response policy for agent
        `i`, given that the other agents are playing policies in `policy`. We
        can solve the MDP exactly using one of our methods from chapter 7.
        """
        def T_prime(s, a_i, s_prime): return self.transition(s, joint_policy(policy, SimpleGamePolicy(a_i), i), s_prime)
        def R_prime(s, a_i): return self.reward(s, joint_policy(policy, SimpleGamePolicy(a_i), i), i)
        policy_i = M.solve(MDP(self.gamma, self.S, self.A[i], T_prime, R_prime))
        return MGPolicy({s: SimpleGamePolicy(policy_i(s)) for s in self.S})

    def softmax_response(self, policy: TODO, i: int, lam: float, M: ExactSolutionMethod = LinearProgramFormulation()) -> 'MGPolicy':
        """The softmax response of agent `i` to join polocu `policy` with precision parameter `lam`"""
        def T_prime(s, a_i, s_prime): return self.transition(s, joint_policy(policy, SimpleGamePolicy(a_i), i), s_prime)
        def R_prime(s, a_i): return self.reward(s, joint_policy(policy, SimpleGamePolicy(a_i), i), i)
        mdp = MDP(self.gamma, self.S, joint_action(self.A), T_prime, R_prime)
        policy_i = M.solve(mdp)
        def Q(s, a): return mdp.lookahead(policy_i.U, s, a)
        def p(s): return SimpleGamePolicy({a: np.exp(lam * Q(s, a)) for a in self.A[i]})
        return MGPolicy({s: p(s) for s in self.S})

    def randstep(self, s: Any, a: Any) -> tuple[Any, float]:
        """Function for taking a random step"""
        s_prime = random.choices(self.S, weights=[self.T(s, a, s_prime) for s_prime in self.S])[0]
        r = self.R(s, a)
        return s_prime, r

    def simulate(self, policy: list['MGSimulativeMethod'], k_max: int, b: np.ndarray) -> list['MGSimulativeMethod']:
        """Simulates the joint policy `policy` for `k_max` steps starting from a state randomly sampled from `b`"""
        # NOTE: For now, operating under assumption that the belief distribution is categorical.
        # TODO: Might make `b` callable in the future
        s = random.choices(self.S, weights=b)[0]
        for _ in range(k_max):
            a = tuple[(policy_i(s=s)() for policy_i in policy)]
            s_prime, r = self.randstep(s, a)
            for policy_i in policy:
                policy_i.update(s, a, s_prime)
            s = s_prime
        return policy


class MGPolicy():  # TODO - Determine if it should inherit from SimpleGamePolicy or not
    """
    An MG policy is a mapping from states to simple game policies, introduced in
    the previous chapter. We can construct it by passing in a generator to
    construct the dictionary. The probability that a policy (either for an MG or
    a simple game) assigns to taking action `a_i` from state `s` is `policy(a_i, s)`.
    """
    def __init__(self, p: dict[Any, SimpleGamePolicy]):
        self.p = p  # dictionary mapping states to simple game policies

    def __call__(self, a_i: Any, s: Any) -> float:
        return self.p[s](a_i)


class MGNashEquilibrium():
    """This nonlinear program computes a Nash Equilibrium for an MG `P`."""
    def solve(self, P: MG) -> list[MGPolicy]:
        I, S, A, R, T = P.tensorform()
        gamma = P.gamma
        joint_A = joint_action(A)
        opti = casadi.Opti()

        # Variables
        U = opti.Variable(len(I), len(S))
        policy = [[opti.Variable(len(A[i])) for s in S] for i in I]

        # Bounds
        opti.subject_to([policy[i, s] >= 0 for i in I for s in S])

        # Nonlinear objective
        objective = sum([U[i, s] - sum([casadi.mtimes([policy[j][s][a[j]] for j in I])\
                         * (R[s, y][i] + gamma * sum([T[s, y, s_prime] * U[i, s_prime] for s_prime in S]))\
                         for y, a in enumerate(joint_A)]) for i in I for s in S])
        opti.minimize(objective)

        # Constraints
        opti.subject_to([
            U[i, s] >= sum([
                casadi.mtimes([(1.0 if a[j] == a_i else 0.0) if j == i else policy[j][s][a[j]] for j in I])\
                * (R[s, y][i] + gamma * sum([T[s, y, s_prime] * U[i, s_prime] for s_prime in S]))
                for y, a in enumerate(joint_A)])
            for i in I for s in S for a_i in A[i]])
        opti.subject_to([sum([policy[i][s][a_i] for a_i in A[i]]) == 1 for i in S for s in S])
        
        # Utilize the Ipopt solver
        options = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        opti.solver("ipopt", options)

        # Optimize
        sol = opti.solve()

        # Return Nash Equilibrium optimal policies
        policy_opt = [[sol.value(policy[i][s]) for s in S] for i in I]
        def policy_i_opt_sg(i, s): return SimpleGamePolicy({P.A[i][a_i]: policy_opt[i][s][a_i] for a_i in A[i]})
        def policy_i_opt(i): return MGPolicy({P.S[s]: policy_i_opt_sg(i, s) for s in S})
        return [policy_i_opt(i) for i in I]


class MGSimulativeMethod(ABC):
    def __init__(self, P: MG, i: int):
        self.P = P                # simple game
        self.i = i                # agent index

    @staticmethod
    @abstractmethod
    def create_from_game(P: MG, i: int) -> 'MGSimulativeMethod':
        pass

    def __call__(self, s: Any = None) -> SimpleGamePolicy:
        pass

    @abstractmethod
    def update(self, s: Any, a: list[Any], s_prime: Any):
        pass

# TODO - Evaluate if a is `Any` or should be `tuple[Any]` if representing joint action space

class MGFictitiousPlay(MGSimulativeMethod):
    def __init__(self, P: MG, i: int, Qi: dict[tuple[Any, Any], float], Ni: dict[tuple[Any, Any, Any], int]):
        super().__init__(P, i)
        self.Qi = Qi  # state-action value estimates
        self.Ni = Ni  # state-action counts

    @staticmethod
    def create_from_game(P: MG, i: int) -> 'MGFictitiousPlay':
        Qi = {(s, a): P.R(s, a)[i] for s in P.S for a in joint_action(P.A)}
        Ni = {(j, s, a_j): 1 for j in P.I for s in P.S for a_j in P.A[j]}
        return MGFictitiousPlay(P, i, Qi, Ni)

    def __call__(self, s: Any) -> SimpleGamePolicy:
        pass  # TODO

    def update(self, s: Any, a: list[Any], s_prime: Any):
        pass  # TODO

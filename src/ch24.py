"""Chapter 24: Multiagent Reasoning"""

import casadi
import cvxpy as cp
import numpy as np
import random
import warnings

from abc import ABC, abstractmethod
from itertools import product
from typing import Any, Callable, Union

from ch23 import project_to_simplex

warnings.simplefilter(action='ignore', category=FutureWarning)


def joint_action(X: list[list[Any]]) -> list[tuple[Any]]:
    """We can use `joint_action(A)` to construct the joint action space from `A`.""" 
    return list(product(*X))


def joint_policy(policy: list['SimpleGamePolicy'],    # policies of other agents
                 policy_i: 'SimpleGamePolicy',        # policy of agent i
                 i: int                                   # i
                 ) -> list['SimpleGamePolicy']:
    return [policy_i if i == j else policy_j for (j, policy_j) in enumerate(policy)]


class SimpleGame():
    """Data structure of a simple game"""
    def __init__(self, gamma: float, I: list[Any], A: list[list[Any]], R: Callable[[Any, Any], np.ndarray]):
        self.gamma = gamma  # discount factor
        self.I = I          # agents
        self.A = A          # joint action space
        self.R = R          # joint reward function

    def tensorform(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        I_tensor = np.arange(len(self.I))
        A_tensor = np.array([np.arange(len(self.A[i])) for i in range(len(self.I))])
        R_tensor = np.array([self.R(a) for a in joint_action(self.A)])
        return I_tensor, A_tensor, R_tensor

    def utility(self, policy: list['SimpleGamePolicy'], i: int) -> float:
        """
        We can use `utility(policy, i)` to compute the utility associated with
        executing joint policy `policy` in the game from the perspective of
        agent `i`.
        """
        def p(a): return np.prod([policy_j(a_j) for (policy_j, a_j) in zip(policy, a)])
        return np.sum([self.R(a)[i]*p(a) for a in joint_action(self.A)])

    def best_response(self, policy: list['SimpleGamePolicy'], i: int) -> 'SimpleGamePolicy':
        """
        For a simple game, we can compute a deterministic best response for agent
        `i`, given that the other agents are playing the policies in `policy`.
        
        NOTE: I added non-determinism in the case where multiple actions have
        the same utility.
        """
        def U(a_i): return self.utility(joint_policy(policy, SimpleGamePolicy(a_i), i), i)
        U_i = np.array([U(a) for a in self.A[i]])
        options = np.argwhere(U_i == np.max(U_i)).T[0]
        selected = options[0] if len(options) == 1 else np.random.choice(options)
        a_i = self.A[i][selected]
        return SimpleGamePolicy(a_i)

    def softmax_response(self, policy: list['SimpleGamePolicy'], i: int, lam: float) -> 'SimpleGamePolicy':
        """
        For a simple game and a particular agent `i`, we can compute the softmax
        response policy `policy_i`, given that the other agents are playing the
        policies in `policy`. This computation requires specifying the precision
        parameter `lam`.
        """
        A_i = self.A[i]
        def U(a_i): return self.utility(joint_policy(policy, SimpleGamePolicy(a_i), i), i)
        return SimpleGamePolicy({a_i: np.exp(lam * U(a_i)) for a_i in A_i})

    def simulate(self, policy: list['SimpleGameSimulativeMethod'], k_max: int) -> list['SimpleGameSimulativeMethod']:
        """
        A simulation of a joint policy in a simple game for `k_max` iterations.
        The joint policy `policy` is a vector of policies that can be individually
        updated through calls to `policy_i.update(a)`.
        """
        for _ in range(k_max):
            a = [policy_i() for policy_i in policy]
            for policy_i in policy:
                policy_i.update(a)
        return policy


class TODO():  # Generic for class typing that needs to be clarified
    pass

# TODO - Also need to think deeper about best structure for how SimpleGamePolicy and MGPolicy interact

class SimpleGamePolicy():
    """
    A policy associated with an agent is represented by a dictionary that maps
    actions to probabilities. There are different ways to construct a policy. One
    way is to pass in a dictionary directory, in which case the probabilities are
    normalized. We can also construct a policy by passing in an action, in which
    case it assigns probability 1 to that action.
    """
    def __init__(self, p: dict[Any, float] | Any):
        # self.p: dictionary mapping actions to probabilities
        if isinstance(p, dict):  # p: dictionary mapping actions to unnormalized probs
            val_sum = np.sum(list(p.values()))
            self.p = {k: v / val_sum for k, v in p.items()}
        else:                    # p: a single action, assigned probability 1
            self.p = {p: 1.0}

    def __call__(self, a_i: Any = None, s: TODO = None) -> float:
        """
        If we have an individual policy `policy_i`, we can call `policy_i(a_i)`
        to compute the probability the policy associates with action `a_i`. If
        we call `policy_i()`, then it will return a random action according to
        that policy.
        """
        if a_i is not None:
            return self.p.get(a_i, 0.0)
        return random.choices(list(self.p.keys()), weights=list(self.p.values()))[0]


class SimpleGamePolicySolutionMethod(ABC):
    @abstractmethod
    def solve(self, P: SimpleGame) -> Union[list[SimpleGamePolicy], 'JointCorrelatedPolicy']:
        pass


class NashEquilibrium(SimpleGamePolicySolutionMethod):
    """
    This nonlinear program computes a Nash equilibrium for a simple game `P`.
    """
    def solve(self, P: SimpleGame) -> list[SimpleGamePolicy]:
        I, A, R = P.tensorform()
        joint_A = joint_action(A)
        opti = casadi.Opti()

        # Variables
        U = opti.variable(len(I))
        policy = [opti.variable(len(A[i])) for i in I]

        # Bounds
        opti.subject_to([policy[i] >= 0 for i in I])

        # Nonlinear objective
        objective = sum([U[i] - sum([casadi.mtimes([policy[j][a[j]] for j in I]) * R[y, i] \
                         for (y, a) in enumerate(joint_A)]) for i in I])
        opti.minimize(objective)

        # Constraints
        opti.subject_to([
            U[i] >= sum([
                casadi.mtimes([(1.0 if a[j] == a_i else 0.0) if j == i else policy[j][a[j]] for j in I])\
                * R[y, i] for (y, a) in enumerate(joint_A)])\
            for i in I for a_i in A[i]])
        opti.subject_to([sum([policy[i][a_i] for a_i in A[i]]) == 1 for i in I])

        # Utilize the Ipopt solver
        options = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        opti.solver("ipopt", options)

        # Optimize
        sol = opti.solve()

        # Return Nash Equilibrium optimal policies
        def policy_opt(i): return SimpleGamePolicy({P.A[i][a_i]: sol.value(policy[i][a_i]) for a_i in A[i]})
        return [policy_opt(i) for i in I]


class JointCorrelatedPolicy(SimpleGamePolicy):
    """
    A joint correlated policy is represented by a dictionary that maps joint
    actions to probabilities. If `policy` is a joint correlated policy,
    evaluating `policy(a)` will return the probability associated with the joint
    action `a`.
    """
    def __init__(self, p: dict[tuple[Any], float]):
        super().__init__(p)  # dictionary mapping from joint actions to probabilities


class CorrelatedEquilibrium(SimpleGamePolicySolutionMethod):
    """
    Correlated equilibria are a more general notion of optimality for a simple
    game `P` than a Nash equilibrium. They can be computed using a linear
    program. The resulting policies are correlated, meaning that the agents
    stochastically select their joint actions.
    """
    def solve(self, P: SimpleGame) -> JointCorrelatedPolicy:
        I, A, R = P.I, P.A, P.R
        joint_A = joint_action(A)
        R_A = np.array([R(a) for a in joint_A]).T
        policy = cp.Variable(len(joint_A))
        objective = cp.Maximize(cp.sum(R_A @ policy))
        constraints = [policy >= 0]  # Nonnegative constraint
        constraints += [cp.sum([R(a)[i] * policy[j] for j, a in enumerate(joint_A) if a[i] == a_i])\
                        >= cp.sum([R(joint_policy(a, a_i_prime, i))[i] * policy[j] for j, a in enumerate(joint_A) if a[i] == a_i]) 
                        for i in I for a_i in A[i] for a_i_prime in A[i]]
        constraints += [cp.sum(policy) == 1]  # Probability constraint
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return JointCorrelatedPolicy({a: policy.value[i] for i, a in enumerate(joint_A)})


class IteratedBestResponse(SimpleGamePolicySolutionMethod):
    """
    Iterated best response involves cycling through the agents and applying
    their best response to the other agents. The algorithm starts with some
    initial policy and stops after `k_max` iterations.
    """
    def __init__(self, k_max: int, initial_policy: list[SimpleGamePolicy]):
        self.k_max = k_max                    # number of iterations
        self.initial_policy = initial_policy

    @staticmethod
    def create_from_game(P: SimpleGame, k_max: int) -> 'IteratedBestResponse':
        """
        For convenience, we have a constructor that takes as input a simple game
        and creates an initial policy that has each agent select actions
        uniformly at random.
        """
        random_policy = [SimpleGamePolicy({a_i: 1.0 for a_i in A_i}) for A_i in P.A]
        return IteratedBestResponse(k_max, random_policy)

    def solve(self, P: SimpleGame) -> list[SimpleGamePolicy]:
        """
        The same solve function will be reused in the next chapter in the context
        of more complicated forms of games.
        """
        policy = self.initial_policy
        for _ in range(self.k_max):
            policy = [P.best_response(policy, i) for i in P.I]
        return policy


class HierarchicalSoftmax(SimpleGamePolicySolutionMethod):
    """The hierarchical softmax model with precision parameter `lam` and level `k`."""
    def __init__(self, lam: float, k: int, initial_policy: list[SimpleGamePolicy]):
        self.lam = lam  # precision parameter
        self.k = k      # level
        self.initial_policy = initial_policy
    
    @staticmethod
    def create_from_game(P: SimpleGame, lam: float, k: int) -> 'HierarchicalSoftmax':
        """
        By default, it starts with an initial joint policy that assigns uniform
        probability to all individual actions.
        """
        random_policy = [SimpleGamePolicy({a_i: 1.0 for a_i in A_i}) for A_i in P.A]
        return HierarchicalSoftmax(lam, k, random_policy)

    def solve(self, P: SimpleGame) -> list[SimpleGamePolicy]:
        policy = self.initial_policy
        for _ in range(self.k):
            policy = [P.softmax_response(policy, i, self.lam) for i in P.I]
        return policy


class SimpleGameSimulativeMethod(ABC):
    def __init__(self, P: SimpleGame, i: int, policy_i: SimpleGamePolicy):
        self.P = P                # simple game
        self.i = i                # agent index
        self.policy_i = policy_i  # current policy

    @staticmethod
    @abstractmethod
    def create_from_game(P: SimpleGame, i: int) -> 'SimpleGameSimulativeMethod':
        pass

    def __call__(self, a_i: Any = None):
        return self.policy_i(a_i)

    @abstractmethod
    def update(self, a: list[Any]):
        pass

# TODO - Evaluate if a is `Any` or should be `tuple[Any]` if representing joint action space

class FictitiousPlay(SimpleGameSimulativeMethod):
    """
    Fictitious play is a simple learning algorithm for an agent `i` of a simple
    game `P` that maintains counts (`N`) of other agent action selections over
    time and averages them, assuming that this is their stochastic policy. It
    then computes a best response to this policy and performs the corresponding
    utility-maximizing action.
    """
    def __init__(self, P: SimpleGame, i: int, N: list[dict[Any, int]], policy_i: SimpleGamePolicy):
        super().__init__(P, i, policy_i)
        self.N = N  # array of action count dictionaries

    @staticmethod
    def create_from_game(P: SimpleGame, i: int) -> 'FictitiousPlay':
        N = [{a_j: 1 for a_j in P.A[j]} for j in P.I]
        uniform = SimpleGamePolicy({a_i: 1.0 for a_i in P.A[i]})
        return FictitiousPlay(P, i, N, uniform)
    
    def update(self, a: list[Any]):
        for (j, a_j) in enumerate(a):
            self.N[j][a_j] += 1
        def p(j): return SimpleGamePolicy({a_j: u/np.sum(list(self.N[j].values())) for (a_j, u) in self.N[j].items()})
        policy = [p(j) for j in self.P.I]
        self.policy_i = self.P.best_response(policy, self.i)


class GradientAscent(SimpleGameSimulativeMethod):
    """
    An implementation of gradient ascent for an agent `i` of a simple game `P`.
    The algorithm updates its distribution over actions incrementally following
    gradient ascent to improve the expected utility. The projection function
    from Chapter 23 (Algorithm 23.6) is used to ensure that the resulting policy
    remains a valid probability distribution.
    """
    def __init__(self, P: SimpleGame, i: int, t: int, policy_i: SimpleGamePolicy):
        super().__init__(P, i, policy_i)
        self.t = t  # time step

    @staticmethod
    def create_from_game(P: SimpleGame, i: int) -> 'GradientAscent':
        uniform = SimpleGamePolicy({a_i: 1.0 for a_i in P.A[i]})
        return GradientAscent(P, i, 1, uniform)

    def update(self, a: list[Any]):
        A_i = self.P.A[self.i]
        def joint_p(a_i): return [SimpleGamePolicy(a_i if j == self.i else a[j]) for j in self.P.I]
        r = np.array([self.P.utility(joint_p(a_i), self.i) for a_i in A_i])
        policy_prime = np.array([self.policy_i(a_i) for a_i in A_i])
        policy = project_to_simplex(policy_prime + r/np.sqrt(self.t))
        self.t += 1
        self.policy_i = SimpleGamePolicy({a_i: p for a_i, p in zip(A_i, policy)})

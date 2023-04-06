import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Callable

from ch07 import MDP


class ActorCriticMethod(ABC):
    @abstractmethod
    def gradient(self, policy: Callable[[np.ndarray, Any], Any],
                 theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass


class BasicActorCritic(ActorCriticMethod):
    pass  # TODO


class GeneralizedAdvantageEstimation(ActorCriticMethod):
    pass  # TODO


class DeterministicPolicyGradient(ActorCriticMethod):
    pass  # TODO

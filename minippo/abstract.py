import abc
from typing import Any, Dict, Generic, Optional

from torch import Tensor

from .data import Action, ActType, ExperienceBuffer, LearningBatch


class ExperienceReplayInterface(Generic[ActType], abc.ABC):
    @abc.abstractmethod
    def incorporate(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_learning_batch(self, batch_size: int) -> LearningBatch:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass


class AlgoWorkerInterface(Generic[ActType], abc.ABC):
    @abc.abstractmethod
    def sample_action(self, observation: Tensor) -> Action[ActType]:
        pass


class AlgoLearnerInterface(Generic[ActType], abc.ABC):
    @abc.abstractmethod
    def get_worker(
        self, params: Optional[Dict[str, Any]] = None
    ) -> AlgoWorkerInterface[ActType]:
        pass

    @abc.abstractmethod
    def incorporate_experience_buffer(self, buffer: ExperienceBuffer) -> None:
        pass

    @abc.abstractmethod
    def fit(self) -> Dict[str, float]:
        pass

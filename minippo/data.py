import dataclasses
from typing import List, Optional, TypeVar, Generic

import torch
from torch import Tensor
from torch.distributions import Distribution


ActType = TypeVar("ActType")


@dataclasses.dataclass
class Action(Generic[ActType]):
    action: Tensor
    log_prob: Optional[Tensor]

    @classmethod
    def from_distribution(cls, distribution: Distribution):
        action = distribution.sample()
        return cls(action=action, log_prob=distribution.log_prob(action))

    def adapt_action(self) -> List[ActType]:
        return []


@dataclasses.dataclass
class ExperienceItem(Generic[ActType]):
    step: int
    observation: Tensor
    action: Action[ActType]
    reward: float
    terminated: bool
    truncated: bool
    observation_next: Optional[Tensor] = None


@dataclasses.dataclass
class ExperienceBuffer(Generic[ActType]):
    observations: List[Tensor] = dataclasses.field(default_factory=list)
    actions: List[Action[ActType]] = dataclasses.field(default_factory=list)
    rewards: List[float] = dataclasses.field(default_factory=list)
    termination_flags: List[bool] = dataclasses.field(default_factory=list)
    truncation_flags: List[bool] = dataclasses.field(default_factory=list)
    observations_next: List[Tensor] = dataclasses.field(default_factory=list)
    finalized: bool = False

    def save(self, exp: ExperienceItem[ActType]):
        if exp.observation_next is None:
            raise RuntimeError("Unfinalized experience item received for saving.")
        if self.finalized:
            raise RuntimeError("Cannot save into a finalized ExperienceBuffer.")
        self.observations.append(exp.observation)
        self.observations_next.append(exp.observation_next)
        self.actions.append(exp.action)
        self.rewards.append(exp.reward)
        self.termination_flags.append(exp.terminated)
        self.truncation_flags.append(exp.truncated)
        if exp.terminated or exp.truncated:
            self.finalized = True

    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.termination_flags = []
        self.truncation_flags = []
        self.finalized = False


@dataclasses.dataclass
class LearningBatch:
    observation: Tensor
    action: Tensor
    action_log_prob: Optional[Tensor]
    reward: Tensor
    observation_next: Tensor


@dataclasses.dataclass
class AdvantageEstimationResult:
    returns: Tensor
    advantages: Tensor


def discount_rewards(
    rewards: Tensor,
    termination_flags: Tensor,
    gamma: float,
) -> AdvantageEstimationResult:
    returns = torch.zeros_like(rewards)
    cumulative_sum = torch.zeros(1)
    for i in range(len(rewards) - 1, -1, -1):
        cumulative_sum *= (1 - termination_flags[i]) * gamma
        cumulative_sum += rewards[i]
        returns[i] = cumulative_sum
    advantages = normalize(returns, dim=0)
    return AdvantageEstimationResult(returns, advantages)


def generalized_advantage_estimation(
    rewards: Tensor,
    values: Tensor,
    values_next: Tensor,
    termination_flags: Tensor,
    lmbda: float,
    gamma: float,
) -> AdvantageEstimationResult:
    delta = rewards + gamma * values_next * (1 - termination_flags) - values
    adv_estimation_result = discount_rewards(delta, termination_flags, gamma=gamma * lmbda)
    adv_estimation_result.advantages = adv_estimation_result.returns
    adv_estimation_result.returns = adv_estimation_result.advantages + values
    return adv_estimation_result


def normalize(x: Tensor, dim: int = 0) -> Tensor:
    return (x - x.mean(dim=dim)) / torch.std(x, dim=dim)

import dataclasses
from typing import Generic, TypeVar

import torch
from torch import Tensor
from torch.distributions import Distribution

ActType = TypeVar("ActType")


@dataclasses.dataclass
class Action(Generic[ActType]):
    action: Tensor
    log_prob: Tensor | None

    @classmethod
    def from_distribution(cls, distribution: Distribution):
        action = distribution.sample()
        return cls(action=action, log_prob=distribution.log_prob(action))

    def adapt_action(self) -> list[ActType]:
        return []


@dataclasses.dataclass
class ExperienceBuffer(Generic[ActType]):
    observations: list[Tensor] = dataclasses.field(default_factory=list)
    actions: list[Tensor] = dataclasses.field(default_factory=list)
    rewards: list[Tensor] = dataclasses.field(default_factory=list)
    termination_flags: list[Tensor] = dataclasses.field(default_factory=list)
    truncation_flags: list[Tensor] = dataclasses.field(default_factory=list)
    observations_next: list[Tensor] = dataclasses.field(default_factory=list)

    def save(
        self,
        observation: Tensor,
        action: Tensor,
        reward: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        observation_next: Tensor,
    ):
        self.observations.append(observation)
        self.observations_next.append(observation_next)
        self.actions.append(action)
        self.rewards.append(reward)
        self.termination_flags.append(terminated)
        self.truncation_flags.append(truncated)

    def reset(self):
        for field in dataclasses.fields(self):
            setattr(self, field.name, field.default)


@dataclasses.dataclass
class LearningBatch:
    observation: Tensor
    action: Tensor
    action_log_prob: Tensor | None
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
    advantages = discount_rewards(delta, termination_flags, gamma=gamma * lmbda).returns
    returns = advantages + values
    return AdvantageEstimationResult(
        returns=returns,
        advantages=advantages,
    )


def normalize(x: Tensor, dim: int = 0) -> Tensor:
    return (x - x.mean(dim=dim)) / (torch.std(x, dim=dim, unbiased=False) + 1e-8)
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


@dataclasses.dataclass
class ExperienceBuffer(Generic[ActType]):
    observations: list[Tensor] = dataclasses.field(default_factory=list)
    actions: list[Tensor] = dataclasses.field(default_factory=list)
    action_logprobs: list[Tensor] = dataclasses.field(default_factory=list)
    rewards: list[Tensor] = dataclasses.field(default_factory=list)
    termination_flags: list[Tensor] = dataclasses.field(default_factory=list)
    truncation_flags: list[Tensor] = dataclasses.field(default_factory=list)
    observations_next: list[Tensor] = dataclasses.field(default_factory=list)

    def save(
        self,
        observation: Tensor,
        action: Tensor,
        action_logprobs: Tensor,
        reward: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        observation_next: Tensor,
    ):
        self.observations.append(observation)
        self.observations_next.append(observation_next)
        self.actions.append(action)
        self.action_logprobs.append(action_logprobs)
        self.rewards.append(reward)
        self.termination_flags.append(terminated)
        self.truncation_flags.append(truncated)

    def reset(self):
        for field in dataclasses.fields(self):
            setattr(self, field.name, field.default_factory())


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


def normalize(x: Tensor) -> Tensor:
    return (x - x.mean()) / (torch.std(x, unbiased=False) + 1e-8)


def discount_rewards(
    rewards: Tensor,
    episode_ends: Tensor,
    gamma: float,
) -> AdvantageEstimationResult:
    returns = torch.zeros_like(rewards)
    cumulative_sum = torch.zeros(rewards.shape[1])
    for i in range(len(rewards) - 1, -1, -1):
        cumulative_sum = cumulative_sum * (1 - episode_ends[i]) * gamma + rewards[i]
        returns[i] = cumulative_sum

    advantages = normalize(returns)
    return AdvantageEstimationResult(returns=returns, advantages=advantages)


def generalized_advantage_estimation(
    rewards: Tensor,
    values: Tensor,
    values_next: Tensor,
    termination_flags: Tensor,
    truncation_flags: Tensor,
    lmbda: float,
    gamma: float,
) -> AdvantageEstimationResult:
    episode_ends = torch.clamp(termination_flags + truncation_flags, 0.0, 1.0)
    delta = rewards + gamma * values_next * (1 - termination_flags) - values
    gae_advantages = discount_rewards(delta, episode_ends, gamma=gamma * lmbda).returns
    returns = gae_advantages + values
    normalized_advantages = normalize(gae_advantages)
    return AdvantageEstimationResult(
        returns=returns,
        advantages=normalized_advantages,
    )

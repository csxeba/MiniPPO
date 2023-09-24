from typing import Tuple

import torch
from torch import Tensor


def discount(rewards: Tensor, termination_flags: Tensor, gamma):
    returns = torch.zeros_like(rewards)
    cumulative_sum = torch.zeros(1)
    for i in range(len(rewards) - 1, -1, -1):
        cumulative_sum *= (1 - termination_flags[i]) * gamma
        cumulative_sum += rewards[i]
        returns[i] = cumulative_sum
    return returns


def gae_mine(
    rewards: Tensor,
    values: Tensor,
    values_next: Tensor,
    termination_flags: Tensor,
    lmbda: float,
    gamma: float,
) -> Tuple[Tensor, Tensor]:
    delta = rewards + gamma * values_next * (1 - termination_flags) - values
    advantages = discount_rewards(delta, termination_flags, gamma=gamma * lmbda).returns
    returns = delta + values
    return returns, advantages

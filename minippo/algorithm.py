import copy
import dataclasses
from typing import Dict, Any, NamedTuple, Tuple, Callable, Optional

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Distribution

from . import abstract, network, data
from .abstract import AlgoWorkerInterface
from .data import ActType


@dataclasses.dataclass
class Config:
    observation_space_shape: Tuple[int, ...]
    observation_space_dtype: torch.dtype
    action_space_shape: Tuple[int, ...]
    action_space_dtype: torch.dtype
    discount_factor_gamma: float = 0.99
    gae_lambda: float = 0.97
    entropy_beta: float = 0.0
    clip_epsilon: float = 0.2
    target_kl_divergence: float = 0.01
    actor_num_updates: int = 10
    critic_num_updates: int = 10
    experience_max_size: int = 1000

    @classmethod
    def from_env(cls, env: gym.Env, **kwargs) -> "Config":
        return cls(
            observation_space_shape=env.observation_space.shape,
            observation_space_dtype=env.observation_space.dtype,
            action_space_shape=env.action_space.shape,
            action_space_dtype=env.action_space.dtype,
            **kwargs
        )

    def serialize(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def deserialize(cls, mapping: Dict[str, Any]) -> "Config":
        return cls(**mapping)


class PPOBatch(NamedTuple):
    inputs: Tensor
    actions: Tensor
    advantages: Tensor
    old_log_probs: Tensor
    critic_targets: Tensor


class PolicyGradientExperienceReplay(abstract.ExperienceReplayInterface[int]):

    def __init__(
        self,
        max_len: int,
        observation_shape: Tuple[int],
        action_dtype: ActType,
        gae_lmbda: float,
        gae_gamma: float,
    ):
        self.inputs: Tensor = torch.zeros((0, *observation_shape), dtype=torch.float)
        self.actions: Tensor = torch.zeros(0, dtype=[torch.float32, torch.int64][np.issubdtype(action_dtype, np.integer)])
        self.log_probs: Tensor = torch.zeros(0, dtype=torch.float)
        self.returns: Tensor = torch.zeros(0, dtype=torch.float)
        self.advantages: Tensor = torch.zeros(0, dtype=torch.float)
        self.observation_shape = observation_shape
        self.action_dtype = [torch.float32, torch.int64][np.issubdtype(action_dtype, np.integer)]
        self.gae_lmbda = gae_lmbda
        self.gae_gamma = gae_gamma
        self.max_len = max_len

    def incorporate(self, inputs: Tensor, actions: Tensor, log_probs: Tensor, returns: Tensor, advantages: Tensor):
        self.inputs = torch.cat([self.inputs, inputs], dim=0)
        self.actions = torch.cat([self.actions, actions], dim=0)
        self.log_probs = torch.cat([self.log_probs, log_probs], dim=0)
        self.returns = torch.cat([self.returns, returns], dim=0)
        self.advantages = torch.cat([self.advantages, advantages], dim=0)
        assert len(self.inputs) == len(self.returns)
        if len(self.inputs) > self.max_len:
            self.inputs = self.inputs[-self.max_len:]
            self.returns = self.returns[-self.max_len:]
            self.advantages = self.advantages[-self.max_len:]

    def get_learning_batch(self, batch_size: int) -> PPOBatch:
        if batch_size == -1:
            batch_size = len(self.inputs)
        arg = torch.randint(0, len(self.inputs), size=[batch_size])
        return PPOBatch(
            inputs=self.inputs[arg],
            actions=self.actions[arg],
            advantages=self.advantages[arg],
            old_log_probs=self.log_probs[arg],
            critic_targets=self.returns[arg],
        )

    def reset(self):
        self.inputs: Tensor = torch.zeros((0, *self.observation_shape), dtype=torch.float)
        self.actions: Tensor = torch.zeros(0, dtype=self.action_dtype)
        self.log_probs: Tensor = torch.zeros(0, dtype=torch.float)
        self.returns: Tensor = torch.zeros(0, dtype=torch.float)
        self.advantages: Tensor = torch.zeros(0, dtype=torch.float)


class StochasticPolicyWorker(abstract.AlgoWorkerInterface[int]):

    def __init__(self, actor_network: nn.Module) -> None:
        self.actor = actor_network
        self.is_learning = False

    def sample_action(self, observation: Tensor) -> data.Action[int]:
        self.actor.eval()
        with torch.inference_mode():
            distribution: Distribution = self.actor(observation)
        return data.Action.from_distribution(distribution)


class PolicyGradient(abstract.AlgoLearnerInterface[int]):

    def __init__(
        self,
        ppo_config: Config,
        actor_building_fn: Callable[[], network.Actor[int]],
        actor_optimizer_building_fn: Callable[[network.Actor[int]], torch.optim.Optimizer],
    ):
        self.cfg = ppo_config
        self.actor = actor_building_fn()
        self.actor_optimizer = actor_optimizer_building_fn(self.actor)
        self.actor_building_fn = actor_building_fn
        self.experience_replay = PolicyGradientExperienceReplay(
            ppo_config.experience_max_size,
            ppo_config.observation_space_shape,
            ppo_config.action_space_dtype,
            ppo_config.gae_lambda,
            ppo_config.gae_lambda,
        )

    def get_worker(self, params: Optional[Dict[str, Any]] = None) -> AlgoWorkerInterface[ActType]:
        worker_network = self.actor_building_fn()
        worker_network.load_state_dict(copy.deepcopy(self.actor.state_dict()))
        return StochasticPolicyWorker(worker_network)

    def loss_actor(self, batch: PPOBatch) -> Dict[str, Tensor]:
        distr = self.actor(batch.inputs)
        log_probs = distr.log_prob(batch.actions)
        loss = -torch.mean(log_probs * batch.advantages)
        return {"actor_loss": loss, "entropy": distr.entropy()}

    def incorporate_experience_buffer(self, buffer: data.ExperienceBuffer) -> None:
        xs = torch.stack(buffer.observations + [buffer.observations_next[-1]], dim=0)
        rewards = torch.tensor(buffer.rewards)
        term_flags = torch.tensor(buffer.termination_flags, dtype=torch.float)
        actions = torch.tensor([act.action for act in buffer.actions], dtype=torch.int)
        log_probs = torch.tensor([act.log_prob for act in buffer.actions], dtype=torch.float)
        returns = data.discount_rewards(rewards, term_flags, self.cfg.discount_factor_gamma)
        advantages = data.normalize(returns)
        self.experience_replay.incorporate(xs[:-1], actions, log_probs, returns, advantages)

    def fit(self) -> Dict[str, float]:
        self.actor.train()
        batch = self.experience_replay.get_learning_batch(batch_size=-1)
        result = self.loss_actor(batch)
        actor_loss = result["actor_loss"]
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.experience_replay.reset()
        return {k: torch.mean(v).item() for k, v in result.items()}


# class PPO:
#
#     def __init__(
#         self,
#         actor: network.Actor,
#         critic: network.Critic,
#         actor_optimizer: torch.optim.Optimizer,
#         critic_optimizer: torch.optim.Optimizer,
#         ppo_config: Config,
#     ):
#         self.cfg = ppo_config
#         self.actor = actor
#         self.critic = critic
#         self.actor_optimizer = actor_optimizer
#         self.critic_optimizer = critic_optimizer
#
#     def sample(self, state: Tensor) -> data.Action:
#         self.actor.eval()
#         with torch.inference_mode():
#             distribution: Distribution = self.actor(state[None, :])
#         return data.Action.from_distribution(distribution)
#
#     def loss_critic(self, batch: PPOBatch) -> Dict[str, Tensor]:
#         values = self.critic(batch.inputs)
#         loss = F.mse_loss(values, batch.critic_targets[:, None])
#         return {"critic_loss": loss}
#
#     def loss_actor(self, batch: PPOBatch) -> Dict[str, Tensor]:
#         policy_distr = self.actor(batch.inputs)
#         new_log_probs = policy_distr.log_prob(batch.actions)
#         ratio = torch.exp(new_log_probs - batch.old_log_probs)
#         clipped_ratio = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon)
#         surrogate = torch.min(ratio * batch.advantages, clipped_ratio * batch.advantages)
#         entropy_bonus = self.cfg.entropy_beta * policy_distr.entropy()
#         loss = -torch.mean(surrogate - entropy_bonus)
#         return {"actor_loss": loss}
#
#     def train_epoch(
#         self,
#         experience_replay: ExperienceReplay,
#         critic_batch_size: int,
#         actor_batch_size: int,
#     ) -> Dict[str, float]:
#
#         self.critic.train()
#         critic_loss = torch.zeros(1)
#         for ppo_batch in (experience_replay.sample(critic_batch_size) for _ in range(self.cfg.critic_num_updates)):
#             result = self.loss_critic(ppo_batch)
#             critic_loss += result["critic_loss"]
#
#         self.actor.train()
#         actor_loss = torch.zeros(1)
#         for ppo_batch in (experience_replay.sample(actor_batch_size) for _ in range(self.cfg.actor_num_updates)):
#             result = self.loss_actor(ppo_batch)
#             actor_loss += result["actor_loss"]
#
#         actor_loss = actor_loss / actor_batch_size
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()
#
#         critic_loss = critic_loss / critic_batch_size
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()
#
#         return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

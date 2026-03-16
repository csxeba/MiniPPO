import copy
import dataclasses
from typing import Any, Callable, Generic, NamedTuple, TypeVar

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Distribution, MultivariateNormal

from minippo import abstract, data, utils

DistrType = TypeVar("DistrType", MultivariateNormal, Categorical)
DistrFactory = Callable[[Tensor], DistrType]


@dataclasses.dataclass
class Config:
    n_parallel: int
    observation_space_shape: tuple[int, ...]
    observation_space_dtype: torch.dtype
    action_space_shape: tuple[int, ...]
    action_space_dtype: torch.dtype
    discount_factor_gamma: float = 0.99
    gae_lambda: float = 0.97
    entropy_beta: float = 0.0
    clip_epsilon: float = 0.2
    target_kl_divergence: float = 0.01
    actor_num_updates: int = 10
    critic_num_updates: int = 10
    experience_max_size: int = 1000
    actor_batch_size: int = -1
    critic_batch_size: int = -1

    @classmethod
    def from_env(cls, env: gym.Env, **kwargs) -> "Config":
        return cls(
            observation_space_shape=env.observation_space.shape,
            observation_space_dtype=env.observation_space.dtype,
            action_space_shape=env.action_space.shape,
            action_space_dtype=env.action_space.dtype,
            **kwargs,
        )

    def serialize(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def deserialize(cls, mapping: dict[str, Any]) -> "Config":
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
        observation_shape: tuple[int, ...],
        action_dtype: data.ActType,
        gae_lmbda: float,
        gae_gamma: float,
    ):
        self.inputs: Tensor = torch.zeros((0, *observation_shape), dtype=torch.float)
        self.actions: Tensor = torch.zeros(
            0,
            dtype=[torch.float32, torch.int64][np.issubdtype(action_dtype, np.integer)],
        )
        self.log_probs: Tensor = torch.zeros(0, dtype=torch.float)
        self.returns: Tensor = torch.zeros(0, dtype=torch.float)
        self.advantages: Tensor = torch.zeros(0, dtype=torch.float)
        self.observation_shape = observation_shape
        self.action_dtype = [torch.float32, torch.int64][
            np.issubdtype(action_dtype, np.integer)
        ]
        self.gae_lmbda = gae_lmbda
        self.gae_gamma = gae_gamma
        self.max_len = max_len

    def incorporate(
        self,
        inputs: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        returns: Tensor,
        advantages: Tensor,
    ):
        self.inputs = torch.cat([self.inputs, inputs], dim=0)
        self.actions = torch.cat([self.actions, actions], dim=0)
        self.log_probs = torch.cat([self.log_probs, log_probs], dim=0)
        self.returns = torch.cat([self.returns, returns], dim=0)
        self.advantages = torch.cat([self.advantages, advantages], dim=0)
        assert len(self.inputs) == len(self.returns)
        if len(self.inputs) > self.max_len:
            self.inputs = self.inputs[-self.max_len :]
            self.returns = self.returns[-self.max_len :]
            self.advantages = self.advantages[-self.max_len :]

    def get_learning_batch(self, batch_size: int) -> PPOBatch:
        if batch_size == -1:
            arg = torch.arange(len(self.inputs))
        else:
            arg = torch.randint(0, len(self.inputs), size=[batch_size])
        return PPOBatch(
            inputs=self.inputs[arg],
            actions=self.actions[arg],
            advantages=self.advantages[arg],
            old_log_probs=self.log_probs[arg],
            critic_targets=self.returns[arg],
        )

    def reset(self):
        self.inputs: Tensor = torch.zeros(
            (0, *self.observation_shape), dtype=torch.float
        )
        self.actions: Tensor = torch.zeros(0, dtype=self.action_dtype)
        self.log_probs: Tensor = torch.zeros(0, dtype=torch.float)
        self.returns: Tensor = torch.zeros(0, dtype=torch.float)
        self.advantages: Tensor = torch.zeros(0, dtype=torch.float)


class Actor(nn.Module, Generic[DistrType]):
    def forward(self, inputs: Tensor) -> DistrType:
        raise NotImplementedError

    def __call__(self, inputs: Tensor) -> DistrType:
        return self.forward(inputs)


class Critic(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)


class FFActor(Actor[DistrType]):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hiddens: tuple[int, ...],
        distr_factory: DistrFactory,
    ):
        super().__init__()
        layers = [nn.Linear(in_features, hiddens[0]), nn.ReLU()]
        h1 = hiddens[0]
        for h0, h1 in zip(hiddens[:-1], hiddens[1:]):
            layers.extend(
                [
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ]
            )
        layers.extend(
            [
                nn.Linear(h1, out_features),
            ]
        )
        self.layers = nn.Sequential(*layers)
        self.distr_factory = distr_factory

    def forward(self, inputs: Tensor) -> DistrType:
        policy_output = self.layers(inputs)
        distr = self.distr_factory(policy_output)
        return distr


class FFCritic(Critic):
    def __init__(self, in_features: int, hiddens: tuple[int, ...]):
        super().__init__()
        layers = [nn.Linear(in_features, hiddens[0]), nn.ReLU()]
        h1 = hiddens[0]
        for h0, h1 in zip(hiddens[:-1], hiddens[1:]):
            layers.extend([nn.Linear(h0, h1), nn.ReLU()])
        layers.extend([nn.Linear(h1, 1)])
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        critic_values = self.layers(inputs)
        return critic_values


class StochasticPolicyWorker(abstract.AlgoWorkerInterface[int]):
    def __init__(self, actor_network: nn.Module) -> None:
        self.actor = actor_network
        self.is_learning = False

    def sample_action(self, observation: Tensor) -> data.Action[int]:
        self.actor.eval()
        with torch.inference_mode():
            distribution: Distribution = self.actor(observation)
        return data.Action.from_distribution(distribution)


class A2C(abstract.AlgoLearnerInterface):
    def __init__(
        self,
        actor_building_fn: Callable[[], Actor],
        critic_building_fn: Callable[[], Critic],
        actor_optimizer_building_fn: Callable[[Actor], torch.optim.Optimizer],
        critic_optimizer_building_fn: Callable[[Critic], torch.optim.Optimizer],
        ppo_config: Config,
    ):
        self.actor = actor_building_fn()
        self.actor_optimizer = actor_optimizer_building_fn(self.actor)
        self.actor_building_fn = actor_building_fn
        self.critic = critic_building_fn()
        self.critic_optimizer = critic_optimizer_building_fn(self.critic)
        self.experience_replay = PolicyGradientExperienceReplay(
            max_len=ppo_config.experience_max_size,
            observation_shape=ppo_config.observation_space_shape,
            action_dtype=ppo_config.action_space_dtype,
            gae_lmbda=ppo_config.gae_lambda,
            gae_gamma=ppo_config.gae_lambda,
        )
        self.cfg = ppo_config
        if ppo_config.gae_lambda == 0.0:
            print(" [MiniPPO] - Advantage estimation is done with a Value Baseline")
        else:
            print(" [MiniPPO] - Advantage estimation is done with GAE")

    def get_worker(
        self, params: dict[str, Any] | None = None
    ) -> abstract.AlgoWorkerInterface[abstract.ActType]:
        worker_network = self.actor_building_fn()
        worker_network.load_state_dict(copy.deepcopy(self.actor.state_dict()))
        return StochasticPolicyWorker(worker_network)

    def incorporate_experience_buffer(self, buffer: data.ExperienceBuffer) -> None:
        # Shapes: [n_steps, n_parallel, n_dim]
        xs = torch.stack(buffer.observations + [buffer.observations_next[-1]], dim=0)
        rewards = torch.stack(buffer.rewards)
        term_flags = torch.stack(buffer.termination_flags)
        trunc_flags = torch.stack(buffer.truncation_flags)
        actions = torch.stack(buffer.actions)
        log_probs = torch.stack(buffer.action_logprobs)
        n_steps, n_parallel = xs.shape[:2]
        n_steps -= 1  # correct for final obs
        assert (
            n_parallel == self.cfg.n_parallel
        ), f"{n_steps=} != {self.cfg.n_parallel=}"
        x_dims = list(xs.shape[2:])
        self.critic.eval()
        with torch.inference_mode():
            all_values = self.critic(xs.reshape(-1, *x_dims))[..., 0]
            all_values = all_values.reshape(n_steps + 1, n_parallel)
        if self.cfg.gae_lambda > 0.0:
            advantage_estimate = data.generalized_advantage_estimation(
                rewards=rewards,
                values=all_values[:-1],
                values_next=all_values[1:],
                termination_flags=term_flags,
                truncation_flags=trunc_flags,
                lmbda=self.cfg.gae_lambda,
                gamma=self.cfg.discount_factor_gamma,
            )
        else:
            advantage_estimate = data.discount_rewards(
                rewards,
                term_flags,
                self.cfg.discount_factor_gamma,
            )
            advantage_estimate.advantages -= all_values[1:]
        self.experience_replay.incorporate(
            inputs=xs[:-1].reshape(-1, *x_dims),
            actions=actions.reshape(-1),
            log_probs=log_probs.reshape(-1),
            returns=advantage_estimate.returns.reshape(-1),
            advantages=advantage_estimate.advantages.reshape(-1),
        )

    def loss_critic(self, batch: PPOBatch) -> dict[str, Any]:
        values = self.critic(batch.inputs)[:, 0]
        loss = (values - batch.critic_targets).square().mean()
        return {
            "critic_loss": loss,
            "critic_metrics": {
                "value": values.mean().detach().item(),
            },
        }

    def loss_actor(self, batch: PPOBatch) -> dict[str, Tensor]:
        distr = self.actor(batch.inputs)
        log_probs = distr.log_prob(batch.actions)
        entropy = distr.entropy()
        loss = -torch.mean(log_probs * batch.advantages)
        loss = loss - self.cfg.entropy_beta * entropy.mean()
        return {
            "actor_loss": loss,
            "actor_metrics": {
                "entropy": distr.entropy().mean().item(),
                "kld": (batch.old_log_probs - log_probs).mean().item(),
            },
        }

    def fit(self) -> dict[str, float]:
        self.critic.train()
        metrics = {}
        batch = self.experience_replay.get_learning_batch(self.cfg.critic_batch_size)
        result = self.loss_critic(batch)
        self.critic_optimizer.zero_grad()
        result["critic_loss"].backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        critic_metrics = {"critic": result["critic_loss"].detach().item()}
        critic_metrics.update(result["critic_metrics"])
        metrics.update(critic_metrics)
        self.actor.train()
        result = self.loss_actor(batch)
        self.actor_optimizer.zero_grad()
        result["actor_loss"].backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        actor_metrics = {"actor": result["actor_loss"].detach().item()}
        actor_metrics.update(result["actor_metrics"])
        metrics.update(actor_metrics)

        self.experience_replay.reset()
        return metrics


class PPO(A2C):
    def loss_actor(self, batch: PPOBatch) -> dict[str, Any]:
        policy_distr = self.actor(batch.inputs)
        new_log_probs = policy_distr.log_prob(batch.actions)
        prob_ratio = torch.exp(new_log_probs - batch.old_log_probs)
        clip_adv = (
            torch.clamp(
                prob_ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon
            )
            * batch.advantages
        )

        surrogate = -(torch.min(prob_ratio * batch.advantages, clip_adv)).mean()

        return {
            "actor_loss": surrogate,
            "actor_metrics": {
                "entropy": policy_distr.entropy().mean().item(),
                "kld": (batch.old_log_probs - new_log_probs).mean().item(),
                "clip": (surrogate == clip_adv).detach().float().mean().item(),
            },
        }

    def fit(self) -> dict[str, float]:
        self.critic.train()
        metrics = []
        for i in range(self.cfg.critic_num_updates):
            batch = self.experience_replay.get_learning_batch(
                self.cfg.critic_batch_size
            )
            result = self.loss_critic(batch)
            self.critic_optimizer.zero_grad()
            result["critic_loss"].backward()
            self.critic_optimizer.step()
            critic_metrics = {"critic": result["critic_loss"].detach().item()}
            critic_metrics.update(result["critic_metrics"])
            metrics.append(critic_metrics)

        self.actor.train()
        for i in range(1, self.cfg.actor_num_updates + 1):
            batch = self.experience_replay.get_learning_batch(self.cfg.actor_batch_size)
            result = self.loss_actor(batch)
            self.actor_optimizer.zero_grad()
            result["actor_loss"].backward()
            self.actor_optimizer.step()
            actor_metrics = {"actor": result["actor_loss"].detach().item()}
            if i < self.cfg.actor_num_updates:
                result["actor_metrics"].pop(
                    "kld"
                )  # Only interested in KLD in the last iteration
            actor_metrics.update(result["actor_metrics"])
            metrics.append(actor_metrics)

        self.experience_replay.reset()
        return utils.average_dict_of_floats(metrics)

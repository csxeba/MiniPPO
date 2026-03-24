import dataclasses
from typing import Any
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from minippo import util

def _make_mlp(obs_shape: tuple[int, ...], hiddens: tuple[int, ...], output_shape: int) -> nn.Sequential:
    input_shape = np.prod(obs_shape).item()
    layers = [nn.Linear(input_shape, hiddens[0]), nn.ReLU()]
    for h1, h2 in zip(hiddens[:-1], hiddens[1:]):
        layers.extend([
            nn.Linear(h1, h2),
            nn.ReLU(),
        ])
    layers.append(nn.Linear(hiddens[-1], output_shape))
    mlp = nn.Sequential(*layers)
    return mlp

def calculate_gae(
    rewards: torch.Tensor,  # Shape: (T, n_envs)
    value_preds: torch.Tensor,  # Shape: (T, n_envs)
    masks: torch.Tensor,  # Shape: (T, n_envs)
    gamma: float,
    lam: float,
) -> torch.Tensor:
    T, n_envs = rewards.shape
    advantages = torch.zeros(T, n_envs, device=rewards.device)
    gae = 0.0
    values = value_preds.detach()
    for t in reversed(range(T - 1)):
        td_error = (
                rewards[t] + gamma * masks[t] * values[t + 1] - value_preds[t]
        )
        gae = td_error + gamma * lam * masks[t] * gae
        advantages[t] = gae
    return advantages


@dataclasses.dataclass
class LossOutput:
    metrics: dict[str, float]


class ActorCritic:
    def __init__(
        self,
        device: torch.device,
        config: dict,
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        actor_hiddens = tuple(int(h) for h in config["actor_hiddens"].split("-"))
        critic_hiddens = tuple(int(h) for h in config["critic_hiddens"].split("-"))
        self.critic = _make_mlp(config["obs_shape"], critic_hiddens, 1).to(self.device)
        self.actor = _make_mlp(config["obs_shape"], actor_hiddens, config["action_shape"]).to(self.device)

        # define optimizers for actor and critic
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=config["critic_lr"])
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=config["actor_lr"])

        self.cfg = config

    def update_parameters(
        self, actor_loss: torch.Tensor, critic_loss: torch.Tensor,
    ) -> None:
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    def save(self, to_file: Path, metrics_dict: dict | None):
        state = {
            "config": self.cfg,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.critic_optim.state_dict(),
            "critic_optimizer": self.critic_optim.state_dict(),
            "metrics": metrics_dict or {},
        }
        torch.save(state, to_file)

    def load(self, from_file: Path):
        state = torch.load(from_file)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optim.load_state_dict(state["actor_optimizer"])
        self.critic_optim.load_state_dict(state["critic_optimizer"])

    def get_losses(
        self,
        observations: torch.Tensor,
        action_logits: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ) -> LossOutput:
        raise NotImplementedError


class A2C(ActorCritic):

    def get_losses(
        self,
        observations: torch.Tensor,
        action_logits: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ) -> LossOutput:
        self.actor.train()
        self.critic.train()
        value_preds = self.critic(observations).squeeze(2)
        advantages = calculate_gae(
            rewards,
            value_preds,
            masks,
            self.cfg["gamma"],
            self.cfg["lam"],
        )
        critic_loss = torch.square(advantages).mean()
        action_pd = torch.distributions.Categorical(logits=self.actor(observations))
        actor_loss = -(advantages.detach() * action_pd.log_prob(actions)).mean()
        entropy_bonus = -action_pd.entropy().mean()
        kld = torch.distributions.kl_divergence(
            action_pd,
            torch.distributions.Categorical(logits=action_logits)
        ).mean()
        self.update_parameters(
            actor_loss=actor_loss + self.cfg["ent_coef"] * entropy_bonus,
            critic_loss=critic_loss,
        )
        return LossOutput(
            metrics={
                "actor": actor_loss.detach().cpu().item(),
                "critic": critic_loss.detach().cpu().item(),
                "kld": kld.detach().cpu().item(),
            }
        )


class PPO(ActorCritic):

    def get_losses(
        self,
        observations: torch.Tensor,
        action_logits: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ) -> LossOutput:
        self.actor.eval()
        self.critic.eval()

        ent_coef = self.cfg["ent_coef"]
        ppo_batch_size = self.cfg["ppo_batch_size"]
        ppo_max_updates = self.cfg["ppo_max_updates"]
        clip_ratio = self.cfg["ppo_clip_ratio"]

        time, n_envs = rewards.shape

        with torch.no_grad():
            value_preds = self.critic(observations).squeeze(2)
            advantages = calculate_gae(
                rewards,
                value_preds,
                masks,
                self.cfg["gamma"],
                self.cfg["lam"],
            )

            observations = observations.view(-1, *self.cfg["obs_shape"])
            actions = actions.view(-1)
            old_action_lps = action_log_probs.view(-1)
            old_action_logits = action_logits.view(-1, self.cfg["action_shape"])
            advantages = advantages.view(-1)
            returns = value_preds.view(-1) + advantages
            masks = masks.view(-1)

        batch_stream = util.stream_batched_indices(
            max_index=time * n_envs,
            batch_size=ppo_batch_size,
            shuffle=True,
        )
        actor_loss = 0.
        critic_loss = 0.
        kld = 0.
        entropy = 0.

        self.actor.train()
        self.critic.train()
        for i, batched_indices in enumerate(batch_stream, start=1):
            batched_indices = list(batched_indices)
            batch_obs = observations[batched_indices]
            batch_act = actions[batched_indices]
            batch_masks = masks[batched_indices]
            batch_adv = advantages[batched_indices] * batch_masks
            batch_ret = returns[batched_indices] * batch_masks
            batch_old_log_probs = old_action_lps[batched_indices]

            values = self.critic(batch_obs).squeeze(-1)
            logits = self.actor(batch_obs)

            actor_pd = torch.distributions.Categorical(logits=logits)
            log_probs = actor_pd.log_prob(batch_act)
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surrogate1 = ratio * batch_adv
            surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_adv
            actor_loss = -(torch.min(surrogate1, surrogate2).mean())
            entropy_bonus = -actor_pd.entropy().mean()
            critic_loss = F.mse_loss(values, batch_ret, reduction="mean")

            self.update_parameters(
                actor_loss=actor_loss + self.cfg["ent_coef"] * entropy_bonus,
                critic_loss=critic_loss,
            )
            if i >= ppo_max_updates:
                batch_old_logits = old_action_logits[batched_indices]
                batch_old_pd = torch.distributions.Categorical(logits=batch_old_logits)
                kld = torch.distributions.kl_divergence(batch_old_pd, actor_pd).mean().detach().cpu().item()
                actor_loss = actor_loss.detach().cpu().item()
                critic_loss = critic_loss.detach().cpu().item()
                break
        self.actor.eval()
        self.critic.eval()

        return LossOutput(metrics={
            "actor": actor_loss,
            "critic": critic_loss,
            "kld": kld,
        })

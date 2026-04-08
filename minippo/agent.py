import dataclasses
from pathlib import Path
from typing import Callable, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer

from minippo import util


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


class ActorCriticOutput(NamedTuple):
    action_logit: torch.Tensor
    value: torch.Tensor


class ActorCritic(nn.Module):
    def forward(self, x: torch.Tensor) -> ActorCriticOutput:
        raise NotImplementedError


class PPO(nn.Module):
    def __init__(
        self,
        device: torch.device,
        config: dict,
        actor_critic_fn: Callable[[], ActorCritic],
        actor_critic_optim_fn: Callable[[], Optimizer],
    ) -> None:
        super().__init__()
        self.actor_critic = actor_critic_fn()
        self.actor_critic_opt = actor_critic_optim_fn()
        self.cfg = config

        self.device = device
        self.to(device)

        # Delegate these
        self.train = self.actor_critic.train
        self.eval = self.actor_critic.eval
        self.forward = self.actor_critic.forward

    def save(self, to_file: Path, metrics_dict: dict | None):
        state = {
            "metrics": metrics_dict or {},
        }
        state.update(self.state_dict())
        torch.save(state, to_file)

    def load(self, from_file: Path):
        self.load_state_dict(torch.load(from_file))

    def get_losses(
        self,
        observations: torch.Tensor,
        action_logits: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        valid_transitions: torch.Tensor,
    ) -> LossOutput:
        self.eval()

        ppo_batch_size = self.cfg["ppo_batch_size"]
        ppo_max_updates = self.cfg["ppo_max_updates"]
        clip_ratio = self.cfg["ppo_clip_ratio"]

        time, n_envs = rewards.shape

        with torch.no_grad():
            value_preds = self(observations).value.squeeze(-1)
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
            valid_transitions = valid_transitions.view(-1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        batch_stream = util.stream_batched_indices(
            max_index=time * n_envs,
            batch_size=ppo_batch_size,
            shuffle=True,
        )
        actor_loss = 0.
        critic_loss = 0.
        kld = 0.

        self.train()
        for i, batched_indices in enumerate(batch_stream, start=1):
            batched_indices = list(batched_indices)
            batch_obs = observations[batched_indices]
            batch_act = actions[batched_indices]
            batch_masks = masks[batched_indices]
            batch_adv = advantages[batched_indices] * batch_masks
            batch_ret = returns[batched_indices] * batch_masks
            batch_old_log_probs = old_action_lps[batched_indices]
            batch_valid_transitions = valid_transitions[batched_indices]
            n_valid = batch_valid_transitions.sum()

            ac_out = self(batch_obs)

            actor_pd = torch.distributions.Categorical(logits=ac_out.action_logit)
            log_probs = actor_pd.log_prob(batch_act)
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surrogate1 = ratio * batch_adv
            surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_adv
            actor_loss = -((torch.min(surrogate1, surrogate2)) * batch_valid_transitions).sum() / n_valid
            entropy_bonus = -(actor_pd.entropy() * batch_valid_transitions).sum() / n_valid
            critic_loss = (F.mse_loss(ac_out.value.squeeze(-1), batch_ret, reduction="none") * batch_valid_transitions).sum() / n_valid

            self.actor_critic_opt.zero_grad()
            (
                self.cfg["actor_loss_coef"] * actor_loss +
                self.cfg["critic_loss_coef"] * critic_loss +
                self.cfg["entropy_loss_coef"] * entropy_bonus
            ).backward()
            if self.cfg["clip_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg["clip_grad_norm"])
            self.actor_critic_opt.step()

            if i >= ppo_max_updates:
                batch_old_logits = old_action_logits[batched_indices]
                batch_old_pd = torch.distributions.Categorical(logits=batch_old_logits)
                kld = torch.distributions.kl_divergence(batch_old_pd, actor_pd).mean().detach().cpu().item()
                actor_loss = actor_loss.detach().cpu().item()
                critic_loss = critic_loss.detach().cpu().item()
                break
        self.eval()

        return LossOutput(metrics={
            "actor": actor_loss,
            "critic": critic_loss,
            "kld": kld,
        })

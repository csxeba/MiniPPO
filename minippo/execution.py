import dataclasses
from collections import defaultdict, deque
from typing import Any, Callable, NamedTuple
from pathlib import Path
import itertools

import numpy as np
import gymnasium as gym
import pandas as pd
import torch

from minippo.agent import PPO


@dataclasses.dataclass
class TrainData:
    observations: torch.Tensor
    action_logits: torch.Tensor
    actions: torch.Tensor
    action_log_probs: torch.Tensor
    rewards: torch.Tensor
    masks: torch.Tensor
    valid_transitions: torch.Tensor
    _steps_written: set

    @classmethod
    def make_empty(
        cls,
        training_hps: dict[str, Any],
        device: torch.device,
    ) -> "TrainData":
        n_steps_per_update = training_hps["n_steps_per_update"]
        n_envs = training_hps["n_envs"]
        obs_shape = training_hps["obs_shape"]
        action_shape = training_hps["action_shape"]
        return cls(
            observations=torch.zeros([n_steps_per_update, n_envs] + list(obs_shape), device=device, dtype=torch.float32),
            action_logits=torch.zeros(n_steps_per_update, n_envs, action_shape, device=device, dtype=torch.float32),
            actions=torch.zeros(n_steps_per_update, n_envs, device=device, dtype=torch.float32),
            action_log_probs=torch.zeros(n_steps_per_update, n_envs, device=device, dtype=torch.float32),
            rewards=torch.zeros(n_steps_per_update, n_envs, device=device, dtype=torch.float32),
            masks=torch.zeros(n_steps_per_update, n_envs, device=device, dtype=torch.float32),
            valid_transitions=torch.zeros(n_steps_per_update, n_envs, device=device, dtype=torch.float32),
            _steps_written=set(),
        )

    @property
    def n_steps(self) -> int:
        return self.observations.shape[0]

    @property
    def n_envs(self) -> int:
        return self.observations.shape[1]

    def push(
        self,
        step: int,
        obs: torch.Tensor,
        action_logit: torch.Tensor,
        action: torch.Tensor,
        action_log_prob: torch.Tensor,
        reward: torch.Tensor,
        mask: torch.Tensor,
        valid_transition: torch.Tensor,
    ):
        assert step < self.n_steps
        assert step not in self._steps_written
        self._steps_written.add(step)
        self.observations[step] = obs
        self.action_logits[step] = action_logit
        self.actions[step] = action
        self.action_log_probs[step] = action_log_prob
        self.rewards[step] = torch.tensor(reward, dtype=self.rewards.dtype)
        self.masks[step] = torch.tensor(mask, dtype=self.masks.dtype)
        self.valid_transitions[step] = torch.tensor(valid_transition, dtype=self.valid_transitions.dtype)

    def pull(self) -> dict[str, torch.Tensor]:
        data = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.type == torch.Tensor}
        assert len(steps := set(v.shape[0] for v in data.values())) == 1
        assert len(set(v.shape[1] for v in data.values())) == 1
        assert steps == {max(self._steps_written) + 1}
        return data


class LastState(NamedTuple):
    observation: np.ndarray
    valid_transition: np.ndarray


def collect_train_data(
    agent: PPO,
    train_envs: gym.vector.VectorEnv,
    training_hps: dict[str, Any],
    last_state: LastState,
    device: torch.device,
) -> tuple[TrainData, LastState]:

    # assert train_envs.autoreset_mode == gym.vector.vector_env.AutoresetMode.SAME_STEP

    n_steps_per_update = training_hps["n_steps_per_update"]
    data = TrainData.make_empty(training_hps, device)
    obs, valid_transition = last_state
    for step in range(n_steps_per_update):
        obs = torch.tensor(obs, device=device, dtype=torch.float32)
        action_logits = agent(obs).action_logit  # expand batch dim
        action_pd = torch.distributions.Categorical(logits=action_logits)
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)

        next_obs, rewards, terminated, truncated, infos = train_envs.step(
            actions.cpu().numpy()
        )
        mask = torch.tensor([not term for term in terminated])
        data.push(step, obs, action_logits, actions, action_log_probs, rewards, mask, valid_transition)

        obs = next_obs
        valid_transition = torch.tensor([not (term or trunc) for term, trunc in zip(terminated, truncated)])

    return data, LastState(obs, valid_transition=valid_transition)


def train(
    train_hps: dict[str, Any],
    agent: PPO,
    train_vec_env: gym.vector.VectorEnv,
    eval_env: gym.Env,
    log_root: Path,
    checkpoints_root: Path,
):
    rolling_report = defaultdict(lambda: deque(maxlen=train_hps["report_smoothing_window"]))
    best_reward = -np.inf
    reports_for_saving = []
    log_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    n_updates = train_hps["n_updates"]
    n_envs = train_hps["n_envs"]
    obs, _ = train_vec_env.reset()
    last_state = LastState(
        observation=obs,
        valid_transition=np.ones(n_envs, dtype=bool),
    )
    printed_first = False
    try:
        for sample_phase in range(n_updates):

            # train step
            train_data, last_state = collect_train_data(
                agent=agent,
                train_envs=train_vec_env,
                training_hps=train_hps,
                last_state=last_state,
                device=agent.device,
            )
            if not printed_first:
                print(train_data.observations[0, 0])
                printed_first = True
            # print(train_data.observations[0, 0])

            loss_output = agent.get_losses(**train_data.pull())

            # reporting
            report = loss_output.metrics.copy()
            if eval_env is not None:
                report.update(validation_epoch(agent, eval_env, n_rollouts=train_hps["num_eval_rollouts"]))
            report_header = ["E"] + list(report.keys())
            report_header_str = "|" + " | ".join(f"{element: ^10}" for element in report_header) + " |"
            report["E"] = sample_phase
            reports_for_saving.append(report)
            for k, v in report.items():
                rolling_report[k].append(v)
            E_str = f"{sample_phase+1}/{n_updates}"
            report_str = [f"{E_str: >10}"]
            report_str += [f"{np.mean(v): >10.3f}" for k, v in rolling_report.items() if k != "E"]
            report_str = "\r" + "".join(["|", " | ".join(report_str), " |"])
            if sample_phase % train_hps["report_smoothing_window"] == 0:
                print()
            if sample_phase % (train_hps["report_smoothing_window"]*10) == 0:
                print()
                print(report_header_str)
            print(report_str, end="")

            # checkpointing
            best_save_path = checkpoints_root / "best_agent.pt"
            if report["R"] > best_reward:
                agent.save(best_save_path, metrics_dict=report)
                best_reward = report["R"]
            latest_save_path = checkpoints_root / "latest_agent.pt"
            agent.save(latest_save_path, metrics_dict=report)
    finally:
        pd.DataFrame(rolling_report).to_csv(log_root / "metric_logs.csv")
        print(f"Saved metric logs to {log_root}")


def validation_epoch(
    agent: PPO,
    env,
    n_rollouts: int,
) -> dict[str, float]:
    agent.eval()
    report = defaultdict(list)
    with torch.no_grad(), torch.inference_mode():
        for _ in range(n_rollouts):
            obs, info = env.reset()
            rollout_reward = 0.0
            steps_iter = itertools.count()
            values = 0.
            entropies = 0.
            for _ in steps_iter:
                obs = torch.tensor(obs, device=agent.device)
                act_logits, value = agent(obs[None, ...])
                act_pd = torch.distributions.Categorical(logits=act_logits.squeeze(0))
                act = act_pd.sample()
                entropy = act_pd.entropy()
                obs, rew, term, trunc, info = env.step(
                    act.cpu().numpy().item()
                )
                rollout_reward += rew
                values += value.squeeze().cpu().item()
                entropies += entropy.mean().cpu().item()
                if trunc or term:
                    break
            max_step = next(steps_iter)
            report["S"].append(max_step)
            report["R"].append(rollout_reward)
            report["V"].append(values / max_step)
            report["ent"].append(entropies / max_step)
    return {k: np.mean(v).item() for k, v in report.items()}


@torch.inference_mode
def plot(agent: PPO, env: gym.Env, temperature: float = 1.0):
    agent.eval()
    while 1:
        obs, info = env.reset()
        for _ in itertools.count():
            env.render()
            obs = torch.tensor(obs, device=agent.device)
            act_logits = agent.forward_actor(obs[None, None, ...])[0, 0, ...]  # expand batch and num. envs to 1
            act_logits_tempered = act_logits / temperature
            act_pd = torch.distributions.Categorical(logits=act_logits_tempered)
            act = act_pd.sample()
            obs, *_ = env.step(
                act.cpu().numpy().item()
            )

            obs, reward, terminated, truncated, info = env.step(act.squeeze(-1).cpu().item())
            if terminated or truncated:
                np.save("last_logits.npy", act_logits.numpy())
                np.save("last_obs.npy", obs)
                np.save("last_canvas.npy", env.unwrapped._canvas)
                break


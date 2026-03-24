import dataclasses
from collections import defaultdict, deque
from typing import Any, Callable
from pathlib import Path
import itertools

import numpy as np
import gymnasium as gym
import pandas as pd
import torch

from minippo.agent import ActorCritic, LossOutput


@dataclasses.dataclass
class TrainData:
    observations: torch.Tensor
    action_logits: torch.Tensor
    actions: torch.Tensor
    action_log_probs: torch.Tensor
    rewards: torch.Tensor
    masks: torch.Tensor
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
            _steps_written=set(),
        )

    def push(
        self,
        step: int,
        obs: torch.Tensor,
        action_logit: torch.Tensor,
        action: torch.Tensor,
        action_log_prob: torch.Tensor,
        reward: torch.Tensor,
        mask: torch.Tensor,
    ):
        n_steps_per_update = len(self.observations)
        assert step < n_steps_per_update
        assert step not in self._steps_written
        self._steps_written.add(step)
        self.observations[step] = obs
        self.action_logits[step] = action_logit
        self.actions[step] = action
        self.action_log_probs[step] = action_log_prob
        self.rewards[step] = torch.tensor(reward, dtype=self.rewards.dtype)
        self.masks[step] = torch.tensor(mask, dtype=self.masks.dtype)

    def pull(self) -> dict[str, torch.Tensor]:
        data = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.type == torch.Tensor}
        assert len(steps := set(v.shape[0] for v in data.values())) == 1
        assert len(set(v.shape[1] for v in data.values())) == 1
        assert steps == {max(self._steps_written) + 1}
        return data


def collect_train_data(
    actor: Callable[[torch.Tensor], torch.Tensor],
    train_envs: gym.vector.AsyncVectorEnv,
    training_hps: dict[str, Any],
    device: torch.device,
    obs: np.ndarray,
) -> tuple[TrainData, np.ndarray]:

    assert train_envs.autoreset_mode == gym.vector.vector_env.AutoresetMode.SAME_STEP

    n_steps_per_update = training_hps["n_steps_per_update"]
    data = TrainData.make_empty(training_hps, device)

    for step in range(n_steps_per_update):
        obs = torch.tensor(obs, device=device, dtype=torch.float32)
        action_logits = actor(obs)
        action_pd = torch.distributions.Categorical(logits=action_logits)
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)

        next_obs, rewards, terminated, truncated, infos = train_envs.step(
            actions.cpu().numpy()
        )
        mask = torch.tensor([not term for term in terminated])
        data.push(step, obs, action_logits, actions, action_log_probs, rewards, mask)

        obs = next_obs

    return data, obs


def train(
    train_hps: dict[str, Any],
    agent: ActorCritic,
    train_vec_env: gym.vector.AsyncVectorEnv,
    eval_env: gym.Env,
    log_root: Path,
):
    rolling_report = defaultdict(lambda: deque(maxlen=train_hps["report_smoothing_window"]))
    best_reward = -np.inf
    reports_for_saving = []
    log_root.mkdir(parents=True, exist_ok=True)

    n_updates = train_hps["n_updates"]
    obs, _ = train_vec_env.reset()

    try:
        for sample_phase in range(n_updates):

            # train step
            train_data, obs = collect_train_data(
                actor=agent.actor,
                train_envs=train_vec_env,
                training_hps=train_hps,
                device=agent.device,
                obs=obs,
            )
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
            E_str = f"{sample_phase}/{n_updates}"
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
            best_save_path = Path("checkpoints/best_agent.pt")
            if report["R"] > best_reward:
                agent.save(best_save_path, metrics_dict=report)
                best_reward = report["R"]
            latest_save_path = Path("checkpoints/latest_agent.pt")
            agent.save(latest_save_path, metrics_dict=report)
    finally:
        pd.DataFrame(rolling_report).to_csv(log_root / "metric_logs.csv")
        print(f"Saved metric logs to {log_root}")


def validation_epoch(
    agent: ActorCritic,
    _env,
    n_rollouts: int,
) -> dict[str, float]:
        agent.actor.eval()
        agent.critic.eval()
        report = defaultdict(list)
        with torch.no_grad(), torch.inference_mode():
            for _ in range(n_rollouts):
                obs, info = _env.reset()
                rollout_reward = 0.0
                steps_iter = itertools.count()
                values = 0.
                entropies = 0.
                for _ in steps_iter:
                    obs = torch.tensor(obs, device=agent.device)[None, ...]
                    act_logits = agent.actor(obs)
                    value = agent.critic(obs)
                    act_pd = torch.distributions.Categorical(logits=act_logits)
                    act = act_pd.sample()
                    entropy = act_pd.entropy()
                    obs, rew, term, trunc, info = _env.step(
                        act.cpu().numpy().item()
                    )
                    rollout_reward += rew
                    values += value.squeeze(1).cpu().item()
                    entropies += entropy.mean().cpu().item()
                    if trunc or term:
                        break
                max_step = next(steps_iter)
                report["S"].append(max_step)
                report["R"].append(rollout_reward)
                report["V"].append(values / max_step)
                report["ent"].append(entropies / max_step)
        return {k: np.mean(v).item() for k, v in report.items()}


def plot():
    env = gym.make("LunarLander-v3", render_mode="human")
    if which != "random":
        save_path = Path(f"checkpoints/{which}.pt")
        assert save_path.exists()
        checkpoint = torch.load(save_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = A2C(device, checkpoint["config"])
        agent.load(save_path)

    while 1:
        observation, info = env.reset()
        for _ in itertools.count():
            env.render()
            if which == "random":
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
            else:
                observation = torch.from_numpy(observation).float().to(device).unsqueeze(0)
                actions, action_log_probs, state_value_preds, entropy = agent.select_action(observation)
                observation, reward, terminated, truncated, info = env.step(actions.squeeze(-1).cpu().numpy())
            if terminated or truncated:
                break

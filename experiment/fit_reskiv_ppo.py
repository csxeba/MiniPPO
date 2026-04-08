import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch import nn
import gymnasium as gym

from grund.reskiv.environment import Reskiv, ReskivConfig

from minippo.agent import PPO, ActorCritic, ActorCriticOutput
from minippo.execution import train, plot
from minippo.util import make_mlp


class ObservationTransform(gym.ObservationWrapper):

    def __init__(self, env: Reskiv, max_len: int = 20):
        super().__init__(env)
        self.max_len = max_len
        self.env = env
        self.canvas = np.zeros((max_len, 3), dtype=np.float32)
        self._observation_space = gym.spaces.Box(
            low=np.array(max_len*[[0.0, 0.0, 0.0]], dtype=np.float32),
            high=np.array(max_len*[[1.0, 1.0, 3.0]], dtype=np.float32),
            shape=(max_len, 3),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        assert obs.ndim == 2  # [n_obj, 3: x, y, type]
        obs = obs[:self.max_len]
        num_entities = len(obs)
        obs[..., 2] += 1  # shift the type indicator to house the "padding" type.
        self.canvas[:] = 0
        self.canvas[:num_entities] = obs
        return self.canvas


class DeepSet(ActorCritic):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        proj_hiddens: tuple[int, ...],
        output_hiddens: tuple[int, ...],
        n_actor_outputs: int,
        n_critic_outputs: int = 1,
    ) -> None:
        super().__init__()
        assert len(obs_shape) == 3
        fan_in = 2
        self.backbone = make_mlp(fan_in, proj_hiddens[:-1], fan_out=proj_hiddens[-1])
        h0 = proj_hiddens[-1]*2 + 2 + 2  #  encoded past + encoded present + self xy + target xy
        self.actor = make_mlp(h0, output_hiddens, n_actor_outputs)
        self.critic = make_mlp(h0, output_hiddens, n_critic_outputs)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        *indep_dims, n_frame, n_obj, fan_in = x.shape
        past, present = x[..., 0, :, :], x[..., 1, :, :]
        n_enemy = n_obj - 2
        past_enemy = past[..., 2:, :]  # [*indep, n_enemy, 2]
        present_enemy = present[..., 2:, :]  # [*indep, n_enemy, 2]
        active_enemy_past = (past_enemy[..., 2] > 0).float().unsqueeze(-1)  # [*indep, n_enemy, 1]
        active_enemy_present = (present_enemy[..., 2] > 0).float().unsqueeze(-1)  # [*indep, n_enemy, 1]
        view_enemy_past = self.backbone(past_enemy[..., :2].reshape(-1, 2)).view(*indep_dims, n_enemy, -1)
        view_enemy_present = self.backbone(present_enemy[..., :2].reshape(-1, 2)).view(*indep_dims, n_enemy, -1)
        views = torch.cat([
            present[..., 0, :2],  # self in present: [*indep, 2]
            present[..., 1, :2],  # goal in present: [*indep, 2]
            (view_enemy_past * active_enemy_past).mean(dim=-2),  # [*indep, d]
            (view_enemy_present * active_enemy_present).mean(dim=-2),  # [*indep, d]
        ], dim=-1)  # [*indep, d*2 + 4]
        return views

    def forward(self, x: torch.Tensor) -> ActorCriticOutput:
        *indep_dims, n_frame, n_obj, fan_in = x.shape
        encoded_obs = self.forward_backbone(x)
        fan_in = encoded_obs.shape[-1]
        encoded_obs = encoded_obs.view(-1, fan_in)
        action_logit = self.actor(encoded_obs).view(*indep_dims, -1)
        value = self.critic(encoded_obs).view(*indep_dims, -1)
        return ActorCriticOutput(action_logit, value)


def env_fn(**kwargs):
    env_config = {
        "canvas_shape": (320, 320),
        "observation_type": "coords",
        "time_limit": 1000,
        "step_penalty": -0.1,
        "enemy_pad_length": 15,
        "frame_stack": 2,
        "reward_scaler": 0.1,
    }
    env_config.update(kwargs)
    maxlen = env_config.pop("enemy_pad_length")
    framestack = env_config.pop("frame_stack")
    rwd_scale = env_config.pop("reward_scaler")
    env = Reskiv(ReskivConfig(**env_config))
    env = ObservationTransform(env, max_len=maxlen)
    env = gym.wrappers.FrameStackObservation(env, stack_size=framestack)
    env = gym.wrappers.TransformReward(env, lambda r: r * rwd_scale)
    return env


@dataclasses.dataclass
class AgentHPs:
    obs_shape: tuple[int, ...]
    action_shape: int
    gamma: float
    lam: float
    actor_hiddens: tuple[int, ...]
    critic_hiddens: tuple[int, ...]
    actor_critic_lr: float
    actor_loss_coef: float
    critic_loss_coef: float
    entropy_loss_coef: float
    clip_grad_norm: float
    ppo_batch_size: int
    ppo_max_updates: int
    ppo_clip_ratio: float



env = env_fn()
agent_hps = AgentHPs(
    obs_shape=tuple(env.observation_space.shape),
    action_shape=env.action_space.n,
    gamma=0.99,
    lam=0.95,  # hyperparameter for GAE
    actor_hiddens=(32, 32),
    critic_hiddens=(32, 32),
    actor_critic_lr=3e-4,
    actor_loss_coef=1.0,
    critic_loss_coef=3.0,
    entropy_loss_coef=0.001,  # coefficient for the entropy bonus (to encourage exploration)
    clip_grad_norm=1.0,
    ppo_batch_size=32,
    ppo_max_updates=64,
    ppo_clip_ratio=0.2,
)

training_hps = dict(
    obs_shape=env.observation_space.shape,
    action_shape=env.action_space.n,
    n_envs=32,
    n_updates=5000,
    n_steps_per_update=128,
    num_eval_rollouts=1,
    report_smoothing_window=50,
)

# set the device
use_cuda = False
if use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


def _agent_fn():
    actor_critic = DeepSet(
        obs_shape=agent_hps.obs_shape,
        proj_hiddens=agent_hps.actor_hiddens,
        output_hiddens=agent_hps.actor_hiddens,
        n_actor_outputs=agent_hps.action_shape,
    )
    return PPO(
        device,
        dataclasses.asdict(agent_hps),
        actor_critic_fn=lambda: actor_critic,
        actor_critic_optim_fn=lambda: torch.optim.Adam(actor_critic.parameters(), lr=agent_hps.actor_critic_lr),
    )


LOGS_ROOT = Path("logs/PPO-Reskiv")
CHKP_ROOT = Path("checkpoints/PPO-Reskiv")
LATEST_CHKP = CHKP_ROOT / "latest_agent.pt"
BEST_CHKP = CHKP_ROOT / "best_agent.pt"


def learn():
    agent = _agent_fn()
    env_config = {
        "step_penalty": -0.01,
    }
    train_envs = gym.vector.SyncVectorEnv(
        env_fns=[lambda: env_fn(**env_config) for _ in range(training_hps["n_envs"])],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    train(
        training_hps,
        agent,
        train_envs,
        eval_env=env_fn(),
        log_root=LOGS_ROOT,
        checkpoints_root=CHKP_ROOT,
    )


def watch():
    agent = _agent_fn()
    agent.load(BEST_CHKP)
    plot(agent, env=env_fn(), temperature=0.75)


if __name__ == "__main__":
    learn()

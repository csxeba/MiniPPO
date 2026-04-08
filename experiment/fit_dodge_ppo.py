import copy
import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch import nn
import gymnasium as gym

from grund.dodge.environment import Dodge, DodgeConfig

from minippo.agent import PPO
from minippo.execution import train, plot


class ObservationTransform(gym.ObservationWrapper):

    def __init__(self, env: Dodge, max_len: int = 20):
        super().__init__(env)
        self.max_len = max_len
        self.env = env
        self.canvas = np.zeros((max_len, 5), dtype=np.float32)
        self._observation_space = gym.spaces.Box(
            low=np.array(max_len*[[-2.0, -2.0, -1.0, -1.0, 0.0]]),
            high=np.array(max_len*[[2.0, 2.0, 1.0, 1.0, 1.0]]),
            shape=(max_len, 5),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        assert obs.ndim == 2  # cat([n_obj, 2: x, y], [n_obj, 2: vx, vy])
        num_objs = len(obs) // 2
        positions = obs[:num_objs]
        velocities = obs[num_objs:]  # drop self velocity -> it kept 0 anyway
        self.canvas[:] = 0
        self.canvas[:num_objs, :2] = positions
        self.canvas[:num_objs, 2:4] = velocities
        self.canvas[:num_objs, 4] = 1  # type enemy
        return self.canvas


class DeepSet(nn.Module):

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        proj_hiddens: tuple[int, ...],
        output_hiddens: tuple[int, ...],
        output_shape: int,
    ) -> None:
        super().__init__()
        fan_in = obs_shape[-1]
        proj_mlp_layers = [
            nn.Linear(fan_in, proj_hiddens[0]),
            nn.ReLU(),
            nn.BatchNorm1d(proj_hiddens[0]),
        ]
        h0 = proj_hiddens[0]
        for h1 in proj_hiddens[1:]:
            proj_mlp_layers.append(nn.Linear(h0, h1))
            proj_mlp_layers.append(nn.ReLU())
            proj_mlp_layers.append(nn.BatchNorm1d(h1))
            h0 = h1
        output_layers = [
            nn.Linear(h0+fan_in, output_hiddens[0]),
            nn.ReLU(),
            nn.BatchNorm1d(output_hiddens[0]),
        ]
        h0 = output_hiddens[0]
        for h1 in output_hiddens[1:]:
            output_layers.append(nn.Linear(h0, h1))
            output_layers.append(nn.ReLU())
            output_layers.append(nn.BatchNorm1d(h1))
            h0 = h1
        output_layers.append(nn.Linear(h0, output_shape))

        self.proj_mlp = nn.Sequential(*proj_mlp_layers)
        self.output_mlp = nn.Sequential(*output_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *indep_shapes, n_obj, fan_in = x.shape
        flat_indep_shapes = np.prod(indep_shapes)

        # Process these with independent DeepSet MLPs
        x_self = x[..., 0, :]
        x_enemies = x[..., 1:, :]
        view_enemies = self.proj_mlp(x_enemies.reshape(flat_indep_shapes*(n_obj-1), fan_in)).view(*indep_shapes, n_obj-1, -1)
        view_enemies = view_enemies.mean(dim=-2)  # mean over the objects
        view = torch.cat([x_self, view_enemies], dim=-1)
        fan_in = view.shape[-1]  # figure out the dimensionality
        out = self.output_mlp(view.view(-1, fan_in)).view(*indep_shapes, -1)
        return out


def env_fn(**kwargs):
    env_cfg = dict(
        dt=0.4,
        simulation_width=256,
        simulation_height=256,
        num_enemy_balls=4,
        ball_radius=7,
        ball_velocity=30,
        step_reward=1.0,
        midpoint_distance_reward_coef=0.1,
        time_limit=1000,
    )
    rwd_scale = kwargs.get("reward_scaler", 0.1)
    env = Dodge(DodgeConfig(**env_cfg))
    env = ObservationTransform(env, max_len=env_cfg["num_enemy_balls"]+1)
    env = gym.wrappers.TransformReward(env, lambda r: r * rwd_scale)
    return env


@dataclasses.dataclass
class AgentHPs:
    obs_shape: tuple[int, ...]
    action_shape: int
    gamma: float
    lam: float
    ent_coef: float
    actor_hiddens: tuple[int, ...]
    critic_hiddens: tuple[int, ...]
    actor_lr: float
    critic_lr: float
    ppo_batch_size: int
    ppo_max_updates: int
    ppo_clip_ratio: float



env = env_fn()
agent_hps = AgentHPs(
    obs_shape=tuple(env.observation_space.shape),
    action_shape=env.action_space.n,
    gamma=0.99,
    lam=0.95,  # hyperparameter for GAE
    ent_coef=0.00,  # coefficient for the entropy bonus (to encourage exploration)
    actor_hiddens=(64, 32),
    critic_hiddens=(64, 32),
    actor_lr=3e-4,
    critic_lr=5e-4,
    ppo_batch_size=32,
    ppo_max_updates=64,
    ppo_clip_ratio=0.2,
)

training_hps = dict(
    obs_shape=env.observation_space.shape,
    action_shape=env.action_space.n,
    n_envs=32,
    n_updates=2000,
    n_steps_per_update=128,
    num_eval_rollouts=1,
    report_smoothing_window=20,
)

# set the device
use_cuda = False
if use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


def _agent_fn():
    # init the agent
    actor = DeepSet(
        obs_shape=agent_hps.obs_shape,
        proj_hiddens=agent_hps.actor_hiddens,
        output_hiddens=(32, 16),
        output_shape=agent_hps.action_shape,
    )
    critic = DeepSet(
        obs_shape=agent_hps.obs_shape,
        proj_hiddens=agent_hps.critic_hiddens,
        output_hiddens=(32, 16),
        output_shape=1,
    )
    return PPO(
        device,
        dataclasses.asdict(agent_hps),
        actor_fn=lambda: actor,
        critic_fn=lambda: critic,
        actor_optim_fn=lambda: torch.optim.Adam(actor.parameters(), lr=agent_hps.actor_lr),
        critic_optim_fn=lambda: torch.optim.Adam(critic.parameters(), lr=agent_hps.critic_lr),
    )



def tanitas():

    env_config = {
    }
    train_envs = gym.vector.SyncVectorEnv(
        env_fns=[lambda: env_fn(**env_config) for _ in range(training_hps["n_envs"])],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    train(
        training_hps,
        _agent_fn(),
        train_envs,
        eval_env=env_fn(),
        log_root=Path(f"logs/PPO-{env.__class__.__name__}"),
        checkpoints_root = Path(f"checkpoints/PPO-{env.__class__.__name__}"),
    )


def megnezes():
    agent = _agent_fn()
    agent.load(Path(f"checkpoints/PPO-{env.__class__.__name__}/best_agent.pt"))
    plot(agent, env=env_fn())



if __name__ == "__main__":
    megnezes()

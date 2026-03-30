import dataclasses
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch import optim
import gymnasium as gym

sys.path.append('/data/Prog/PycharmProjects/grund')
from grund.reskiv.environment import Reskiv, ReskivConfig

from minippo.agent import PPO
from minippo.execution import train, plot


class ObservationTransform(gym.ObservationWrapper):

    def __init__(self, env: Reskiv, max_len: int = 20):
        super().__init__(env)
        self.max_len = max_len
        self.env = env
        self.canvas = np.zeros((max_len, 3), dtype=np.float32)
        self._observation_space = gym.spaces.Box(
            low=np.array(max_len*[[-2.0, -2.0, 0]]),
            high=np.array(max_len*[[2.0, 2.0, 2.0]]),
            shape=(max_len, 3),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        assert obs.ndim == 2  # [n_obj, 3: x, y, type]
        num_enemies = len(obs) - 2
        self.canvas[:] = 0
        normed_coords = obs[1:, :2] - obs[:1, :2]  # relative to self, [n_obj - 1, 2]
        self.canvas[0, :2] = normed_coords[0]  # set the goal object's coords (x, y)
        self.canvas[0, 2] = 1  # set the goal object's type
        self.canvas[1:num_enemies+1, :2] = normed_coords[1:, :2][
            np.argsort(np.linalg.norm(normed_coords[1:, :2], axis=1))
        ][:self.max_len-1]  # set the enemy objects' coords
        self.canvas[1:num_enemies+1, 2] = 2  # set the types
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
        assert len(obs_shape) == 2
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
            nn.Linear(h0, output_hiddens[0]),
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
        *dims, fan_in = x.shape
        views = self.proj_mlp(x.view(-1, fan_in)).view(*dims, -1)  # [B, n, in] -> [B, n, out]
        views = views.mean(dim=-2)
        *dims, fan_in = views.shape
        out = self.output_mlp(views.view(-1, fan_in)).view(*dims, -1)
        return out




def env_fn(**kwargs):
    env = Reskiv(ReskivConfig(observation_type="coords"))
    env = ObservationTransform(env, max_len=20)
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
    gamma=0.999,
    lam=0.98,  # hyperparameter for GAE
    ent_coef=0.01,  # coefficient for the entropy bonus (to encourage exploration)
    actor_hiddens=(32, 32),
    critic_hiddens=(32, 32),
    actor_lr=3e-4,  # 0.0001
    critic_lr=5e-4,  # 0.0005
    ppo_batch_size=32,
    ppo_max_updates=32,
    ppo_clip_ratio=0.2,
)

training_hps = dict(
    obs_shape=env.observation_space.shape,
    action_shape=env.action_space.n,
    n_envs=32,
    n_updates=1000,
    n_steps_per_update=128,
    num_eval_rollouts=1,
    report_smoothing_window=10,
)

# set the device
use_cuda = False
if use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


def tanitas():

    # init the agent
    actor = DeepSet(
        obs_shape=agent_hps.obs_shape,
        proj_hiddens=agent_hps.actor_hiddens,
        output_hiddens=agent_hps.actor_hiddens,
        output_shape=agent_hps.action_shape,
    )
    critic = DeepSet(
        obs_shape=agent_hps.obs_shape,
        proj_hiddens=agent_hps.critic_hiddens,
        output_hiddens=agent_hps.critic_hiddens,
        output_shape=1,
    )
    agent = PPO(
        device,
        dataclasses.asdict(agent_hps),
        actor_fn=lambda: actor,
        critic_fn=lambda: critic,
        actor_optim_fn=lambda: torch.optim.Adam(actor.parameters(), lr=agent_hps.actor_lr),
        critic_optim_fn=lambda: torch.optim.Adam(critic.parameters(), lr=agent_hps.critic_lr),
    )
    train_envs = gym.vector.SyncVectorEnv(
        env_fns=[env_fn for _ in range(training_hps["n_envs"])],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    train(
        training_hps,
        agent,
        train_envs,
        eval_env=env_fn(),
        log_root=Path(f"logs/{agent.__class__.__name__}-Reskiv"),
        checkpoints_root = Path(f"checkpoints/{agent.__class__.__name__}-Reskiv"),
    )


def megnezes():
    agent = PPO(device, agent_hps)
    agent.load(Path(f"checkpoints/{agent.__class__.__name__}-Reskiv/best_agent.pt"))
    plot(agent, env=env_fn(render_mode="human"))



if __name__ == "__main__":
    tanitas()

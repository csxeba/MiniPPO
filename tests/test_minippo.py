import random

import gymnasium as gym
import numpy as np
import torch

from minippo.execution import collect_train_data
from minippo.agent import LossOutput


class DummyEnv(gym.Env):
    """
    A simple environment where:
    - State = current step number
    - Reward = current step number
    - Resets after 'max_steps'
    """
    def __init__(self):
        super().__init__()
        self.max_steps = random.randint(10, 30)
        self.current_step = 0
        self.observation_space = gym.spaces.Box(low=10, high=30, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return np.array([float(self.current_step)], dtype=np.float32), {"max_steps": self.max_steps}

    def step(self, action):
        self.current_step += 1
        obs = np.array([float(self.current_step)], dtype=np.float32)
        reward = float(self.current_step)
        terminated = self.current_step >= self.max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}


class DummyActor:

    def __init__(self, single_env):
        self.n_action = single_env.action_space.n

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        # simulate returning logits
        n_envs = observation.shape[0]
        return torch.zeros(n_envs, self.n_action, dtype=torch.float32)


def test_training_step():
    n_steps = 50
    n_envs = 3

    vec_env = gym.vector.AsyncVectorEnv([DummyEnv for _ in range(n_envs)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    actor = DummyActor(DummyEnv())

    obs, info = vec_env.reset()
    train_data, obs = collect_train_data(
        actor=actor,
        train_envs=vec_env,
        training_hps={
            "n_steps_per_update": n_steps,
            "n_envs": n_envs,
            "action_shape": vec_env.single_action_space.n,
            "obs_shape": vec_env.single_observation_space.shape,
        },
        device=torch.device("cpu"),
        obs=obs,
    )

    assert len(train_data.observations) == len(train_data.rewards) == len(train_data.masks) == n_steps
    assert train_data.observations.shape[1] == train_data.rewards.shape[1] == train_data.masks.shape[1] == n_envs

    for env_idx in range(n_envs):
        env_max_steps = info["max_steps"][env_idx]
        env_obs = train_data.observations[:, env_idx]
        env_reward = train_data.rewards[:, env_idx]
        env_mask = train_data.masks[:, env_idx]
        assert all(env_obs) < env_max_steps
        assert all(env_reward) < env_max_steps

        # Check time alignement
        rollout_start = 0
        rollout_ends = list(np.where(env_mask == 0)[0] + 1)
        rollout_ends.append(env_max_steps)
        for rollout_end in rollout_ends:
            # reward should be shifted by 1 compared to obs
            rollout_obs = env_obs[rollout_start:rollout_end].squeeze(1)
            rollout_rwd = env_reward[rollout_start:rollout_end]
            rollout_mask = env_mask[rollout_start:rollout_end]
            rollout_dif = rollout_rwd - rollout_obs
            assert all(rollout_dif == 1)
            rollout_start = rollout_end


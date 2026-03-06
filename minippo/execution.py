from typing import Callable

import torch
import gymnasium as gym

from minippo import abstract, utils, data


def train_vectorenv(
    algo: abstract.AlgoLearnerInterface,
    env_fn: Callable[[], gym.Env],
    num_workers: int,
    num_epochs: int,
    steps_per_update: int,
):
    metrics_aggregator = utils.MetricAggregator()
    experience_buffer = data.ExperienceBuffer()
    vec_env = gym.vector.AsyncVectorEnv([env_fn for _ in range(num_workers)])
    worker = algo.get_worker()
    obs, info = vec_env.reset()
    for episode in range(1, num_epochs + 1):
        for step in range(steps_per_update):
            obs = torch.from_numpy(obs).float()
            action_obj = worker.sample_action(obs)
            obs_next, reward, termination, truncation, info = vec_env.step(action_obj.action.numpy())
            experience_buffer.save(
                observation=obs,
                action=action_obj.action,
                reward=reward,
                terminated=termination,
                truncated=truncation,
                observation_next=obs_next,
            )
            metrics_aggregator.log_metrics({"r": reward})
            obs = obs_next
        metrics = algo.fit()

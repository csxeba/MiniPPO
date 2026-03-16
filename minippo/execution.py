import itertools
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from minippo import abstract, data, utils


def evaluate(
    algo: abstract.AlgoLearnerInterface,
    env_fn: Callable[[], gym.Env],
    num_rollouts: int,
    max_steps: int,
    metrics_aggregator: utils.MetricAggregator | None,
):
    if metrics_aggregator is None:
        metrics_aggregator = utils.MetricAggregator()
    worker = algo.get_worker()
    env = env_fn()
    if max_steps < 1:
        max_steps = float("inf")
    for episode in range(1, num_rollouts + 1):
        obs, info = env.reset()
        rewards = []
        step = 0
        for step in itertools.count(1):
            obs = torch.from_numpy(obs).float()
            action_obj = worker.sample_action(obs.unsqueeze(0))
            obs_next, reward, termination, truncation, info = env.step(
                action_obj.action.numpy().item()
            )
            rewards.append(reward)
            truncation = truncation or (step == max_steps)
            if termination or truncation:
                break
            obs = obs_next
        metrics_aggregator.log_metrics(
            {"r": np.sum(rewards).item(), "steps": float(step)}
        )
    return metrics_aggregator


def train_vectorenv(
    algo: abstract.AlgoLearnerInterface,
    env_fn: Callable[[], gym.Env],
    num_workers: int,
    steps_per_update: int,
    steps_per_epoch: int,
    total_epochs: int,
    smoothing_window_size: int,
):
    experience_buffer = data.ExperienceBuffer()
    vec_env = gym.vector.AsyncVectorEnv([env_fn for _ in range(num_workers)])
    worker = algo.get_worker()
    obs, info = vec_env.reset()
    obs = torch.from_numpy(obs).float()
    metrics_aggregator = utils.MetricAggregator()
    epoch_counter = 0
    epoch_counter_old = -1
    step_counter = 0
    while epoch_counter < total_epochs:
        for step in range(steps_per_update):
            action_obj = worker.sample_action(obs)
            obs_next, reward, termination, truncation, info = vec_env.step(
                action_obj.action.numpy()
            )
            obs_next = torch.from_numpy(obs_next).float()
            reward = torch.from_numpy(reward).float()
            termination = torch.from_numpy(termination).float()
            truncation = torch.from_numpy(truncation).float()
            experience_buffer.save(
                observation=obs,
                action=action_obj.action,
                action_logprobs=action_obj.log_prob,
                reward=reward,
                terminated=termination,
                truncated=truncation,
                observation_next=obs_next,
            )
            obs = obs_next
            step_counter += 1
        algo.incorporate_experience_buffer(experience_buffer)
        metrics = algo.fit()
        metrics_aggregator.log_metrics(metrics)
        epoch_counter = step_counter // steps_per_epoch
        if epoch_counter > epoch_counter_old:
            evaluate(
                algo,
                env_fn,
                num_rollouts=1,
                max_steps=-1,
                metrics_aggregator=metrics_aggregator,
            )
            metrics_aggregator.generate_report(
                epoch_counter, total_epochs, smoothing_window_size
            )
        epoch_counter_old = epoch_counter
        experience_buffer.reset()

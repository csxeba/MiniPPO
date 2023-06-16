import gymnasium as gym
import torch

import minippo as ppo


def env_fn():
    raw_env = gym.make("CartPole-v1")
    obs_wrapped_env = gym.wrappers.NormalizeObservation(gym.wrappers.FlattenObservation(raw_env))
    wrapped_env = gym.wrappers.TimeLimit(obs_wrapped_env, 200)
    return wrapped_env


def get_actor_fn(env: gym.Env[float, int]):
    def f():
        return ppo.network.FFActor(
            in_features=env.observation_space.shape[0],
            out_features=env.action_space.n,
            hiddens=(64, 64),
            distr_factory=lambda logits: torch.distributions.Categorical(logits=logits),
        )
    return f


def get_actor_optimizer_fn(actor: ppo.network.FFActor[int]) -> torch.optim.Optimizer:
    return torch.optim.Adam(actor.parameters(), lr=1e-3)


def main():
    env = env_fn()
    algo = ppo.algorithm.PolicyGradient(
        ppo.algorithm.Config.from_env(env),
        actor_building_fn=get_actor_fn(env),
        actor_optimizer_building_fn=get_actor_optimizer_fn,
    )
    ppo.executions.train_sync(algo, env_fn, num_workers=4, num_epochs=1000, smoothing_window_size=100)


if __name__ == '__main__':
    main()

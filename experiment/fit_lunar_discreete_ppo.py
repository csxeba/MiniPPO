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
            hiddens=[64, 64],
            distr_factory=lambda logits: torch.distributions.Categorical(logits=logits),
        )
    return f


def actor_optimizer_fn(actor: ppo.network.FFActor[int]) -> torch.optim.Optimizer:
    return torch.optim.Adam(actor.parameters(), lr=1e-3)


def get_critic_fn(env: gym.Env[float, int]):
    def f():
        return ppo.network.FFCritic(
            in_features=env.observation_space.shape[0],
            hiddens=[64, 64],
        )
    return f


def critic_optimizer_fn(critic: ppo.network.FFCritic) -> torch.optim.Optimizer:
    return torch.optim.Adam(critic.parameters(), lr=1e-3)


def main():
    env_id = "CartPole-v1"
    env = env_fn()
    algo = ppo.algorithm.PPO(
        actor_building_fn=get_actor_fn(env),
        actor_optimizer_building_fn=actor_optimizer_fn,
        critic_building_fn=get_critic_fn(env),
        critic_optimizer_building_fn=critic_optimizer_fn,
        ppo_config=ppo.algorithm.Config.from_env(
            env,
            critic_num_updates=10,
            actor_num_updates=10,
            critic_batch_size=128,
            actor_batch_size=128,
        )
    )
    ppo.executions.train_sync(algo, env_fn, num_workers=32, num_epochs=1000, smoothing_window_size=100)


if __name__ == '__main__':
    main()

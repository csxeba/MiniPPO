import gymnasium as gym
import numpy as np
import torch

import minippo as ppo


def env_fn():
    raw_env = gym.make("LunarLander-v3")
    # raw_env = gym.make("CartPole-v1")
    wrapped_env = gym.wrappers.NormalizeObservation(
        gym.wrappers.FlattenObservation(raw_env)
    )
    wrapped_env = gym.wrappers.TransformReward(wrapped_env, func=lambda r: r / 100.)
    return wrapped_env


def get_actor_fn(env: gym.Env[float, int]):
    def f():
        return ppo.algorithm.FFActor[torch.distributions.Categorical](
            in_features=env.observation_space.shape[0],
            out_features=env.action_space.n,
            hiddens=(32, 32),
            distr_factory=lambda logits: torch.distributions.Categorical(
                logits=logits, validate_args=False
            ),
        )

    return f


def actor_optimizer_fn(
    actor: ppo.algorithm.FFActor[torch.distributions.Categorical],
) -> torch.optim.Optimizer:
    return torch.optim.Adam(actor.parameters(), lr=1e-4)


def get_critic_fn(env: gym.Env[float, int]):
    def f():
        return ppo.algorithm.FFCritic(
            in_features=env.observation_space.shape[0],
            hiddens=(32, 32),
        )

    return f


def critic_optimizer_fn(critic: ppo.algorithm.FFCritic) -> torch.optim.Optimizer:
    return torch.optim.RMSprop(critic.parameters(), lr=1e-3)


def main():
    env = env_fn()
    N_PARALLEL = 8
    NUM_WORKERS = 32
    STEPS_PER_UPDATE = 32
    algo = ppo.algorithm.A2C(
        actor_building_fn=get_actor_fn(env),
        actor_optimizer_building_fn=actor_optimizer_fn,
        critic_building_fn=get_critic_fn(env),
        critic_optimizer_building_fn=critic_optimizer_fn,
        ppo_config=ppo.algorithm.Config(
            n_parallel=N_PARALLEL,
            observation_space_shape=env.observation_space.shape,
            observation_space_dtype=np.float32,
            action_space_shape=env.action_space.shape,
            action_space_dtype=np.int64,
            discount_factor_gamma=0.999,
            gae_lambda=0.95,
            critic_batch_size=-1,
            actor_batch_size=-1,
            entropy_beta=0.01,
            experience_max_size=NUM_WORKERS * STEPS_PER_UPDATE
        ),
    )
    ppo.execution.train_vectorenv(
        algo,
        env_fn,
        num_workers=N_PARALLEL,
        smoothing_window_size=10,
        steps_per_update=32,
        steps_per_epoch=32*10,
        total_epochs=1000,
    )


if __name__ == "__main__":
    main()

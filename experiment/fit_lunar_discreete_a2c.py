from pathlib import Path

import torch
import gymnasium as gym

from minippo.agent import A2C
from minippo.execution import train

env_name = "LunarLander-v3"


def main():
    env = gym.make(env_name)
    agent_hps = dict(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.n,
        gamma=0.999,
        lam=0.95,  # hyperparameter for GAE
        ent_coef=0.01,  # coefficient for the entropy bonus (to encourage exploration)
        actor_hiddens="32-32",
        critic_hiddens="32-32",
        actor_lr=0.001,
        critic_lr=0.005,
    )
    training_hps = dict(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.n,
        n_envs=10,
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

    # init the agent
    agent = A2C(device, agent_hps)
    train_envs = gym.vector.AsyncVectorEnv(
        env_fns=[lambda: gym.wrappers.TransformReward(gym.make(env_name), lambda r: r / 100) for _ in range(training_hps["n_envs"])],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    train(
        training_hps,
        agent,
        train_envs,
        eval_env=gym.make(env_name),
        log_root=Path("logs/")
    )





if __name__ == "__main__":
    main()

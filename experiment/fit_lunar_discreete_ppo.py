from pathlib import Path

import torch
import gymnasium as gym

from minippo.agent import PPO
from minippo.execution import train, plot

env_name = "CartPole-v1"


def env_fn(**kwargs):
    env = gym.make(env_name, **kwargs)
    env = gym.wrappers.TransformReward(env, lambda r: r / 100)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=300)
    return env


env = env_fn()
agent_hps = dict(
    obs_shape=env.observation_space.shape,
    action_shape=env.action_space.n,
    gamma=0.999,
    lam=0.98,  # hyperparameter for GAE
    ent_coef=0.01,  # coefficient for the entropy bonus (to encourage exploration)
    actor_hiddens="32-32",
    critic_hiddens="32-32",
    actor_lr=3e-4,  # 0.0001
    critic_lr=5e-4,  # 0.0005
    ppo_batch_size=32,
    ppo_max_updates=4,
    ppo_clip_ratio=0.2,
)
training_hps = dict(
    obs_shape=env.observation_space.shape,
    action_shape=env.action_space.n,
    n_envs=32,
    n_updates=100,
    n_steps_per_update=32,
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
    agent = PPO(device, agent_hps)
    train_envs = gym.vector.AsyncVectorEnv(
        env_fns=[env_fn for _ in range(training_hps["n_envs"])],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    train(
        training_hps,
        agent,
        train_envs,
        eval_env=gym.make(env_name),
        log_root=Path(f"logs/{agent.__class__.__name__}-{env_name}"),
        checkpoints_root = Path(f"checkpoints/{agent.__class__.__name__}-{env_name}"),
    )


def megnezes():
    agent = PPO(device, agent_hps)
    agent.load(Path(f"checkpoints/{agent.__class__.__name__}-{env_name}/best_agent.pt"))
    plot(agent, env=env_fn(render_mode="human"))



if __name__ == "__main__":
    megnezes()

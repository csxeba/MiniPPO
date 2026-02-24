import gymnasium as gym
import torch
from torch import Tensor

from minippo import abstract, data, workers


class WorkerMock(abstract.AlgoWorkerInterface):

    def __init__(self, act_space: gym.spaces.Space):
        super().__init__()
        self.act_space = act_space

    def sample_action(self, observation: Tensor) -> data.Action[data.ActType]:
        action = self.act_space.sample()
        action = torch.tensor(action)
        return data.Action(action.unsqueeze(0), log_prob=None)


def main():
    env = gym.make('CartPole-v1')
    worker_future = workers.dispatch(
        worker_fn=lambda: WorkerMock(env.action_space),
        env_fn=lambda: gym.make('CartPole-v1'),
        worker_id=0,
        save_root=None,
    )
    for retval in worker_future:
        print(retval)


if __name__ == '__main__':
    main()

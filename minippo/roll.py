import dataclasses
import itertools
from typing import Any, Dict, Iterable, Optional

import gymnasium
import torch

from .abstract import AlgoWorkerInterface
from .data import ActType, ExperienceItem


@dataclasses.dataclass
class Roll:
    worker: AlgoWorkerInterface
    env: gymnasium.Env

    def synchronized_job(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Iterable[ExperienceItem[ActType]]:
        episode_counter = 1
        while 1:
            observation, info = self.env.reset(seed=seed, options=options)
            for step in itertools.count():
                if not isinstance(observation, torch.Tensor):
                    observation = torch.tensor(observation, dtype=torch.float32)
                action = self.worker.sample_action(observation[None, ...])
                observation_next, reward, termination, truncation, info = self.env.step(
                    action.action[0].item()
                )
                observation_next = torch.tensor(observation_next, dtype=torch.float32)
                experience_item = ExperienceItem(
                    step=step,
                    observation=observation,
                    action=action,
                    reward=float(reward),
                    terminated=termination,
                    truncated=truncation,
                    observation_next=observation_next,
                    episode_id=episode_counter,
                )
                yield experience_item
                if termination or truncation:
                    break
                observation = observation_next
                episode_counter += 1

from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Optional

import torch
from gymnasium import Env

from minippo import abstract, roll
from minippo.data import ExperienceBuffer, ExperienceItem


class WorkerReport(NamedTuple):
    buffer: ExperienceBuffer
    reward_sum: float
    total_steps: int
    worker_id: int

    def to_file(self, path: str) -> None:
        torch.save(
            {
                "reward_sum": self.reward_sum,
                "total_steps": self.total_steps,
                "worker_id": self.worker_id,
                "buffer": self.buffer,
            },
            path,
        )

    @classmethod
    def from_file(cls, path: str) -> "WorkerReport":
        serialized = torch.load(path)
        return cls(
            serialized["buffer"],
            serialized["reward_sum"],
            serialized["total_steps"],
            serialized["worker_id"],
        )


def dispatch(
    worker_fn: Callable[[], abstract.AlgoWorkerInterface],
    env_fn: Callable[[], Env],
    worker_id: int,
    save_root: Optional[Path],
) -> Iterable[ExperienceItem]:
    worker = worker_fn()
    env = env_fn()
    rollout = roll.Roll(worker, env)
    job = rollout.synchronized_job()
    experience: ExperienceItem
    for step, experience in enumerate(job, start=1):
        experience.worker_id = worker_id
        if save_root is not None:
            experience.serialize(
                save_root / f"w{worker_id}_e{experience.episode_id}_s{step}.exp"
            )
        else:
            yield experience

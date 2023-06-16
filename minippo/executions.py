import pickle
import tempfile
from pathlib import Path
from typing import Callable, Optional, Dict, Any, NamedTuple, Union

import numpy as np
import torch
from gymnasium import Env
import torch.multiprocessing as mp

from . import abstract, data, roll, utils


class WorkerReport(NamedTuple):
    buffer: data.ExperienceBuffer
    reward_sum: float
    total_steps: int
    worker_id: int

    def to_file(self, root_path: Union[str, Path]) -> None:
        filename_stub = f"{self.worker_id:0>4}"
        with open(root_path / f"{filename_stub}.meta", "wb") as handle:
            pickle.dump({"reward_sum": self.reward_sum, "total_steps": self.total_steps, "worker_id": self.worker_id}, handle)
        torch.save(self.buffer, root_path / f"{filename_stub}.buf")

    @classmethod
    def from_file(cls, root_path: Union[str, Path], worker_id: int) -> "WorkerReport":
        filename_stub = f"{worker_id:0>4}"
        with open(root_path / f"{filename_stub}.meta", "rb") as handle:
            metadata = pickle.load(handle)
        buffer = torch.load(root_path / f"{filename_stub}.buf")
        return cls(buffer, metadata["reward_sum"], metadata["total_steps"], metadata["worker_id"])


def _dispatch_worker(
    worker_fn: Callable[[], abstract.AlgoWorkerInterface],
    env_fn: Callable[[], Env],
    worker_id: int,
    save_root: Optional[Path] = None,
) -> WorkerReport:
    worker = worker_fn()
    env = env_fn()
    experience_buffer = data.ExperienceBuffer()
    rollout = roll.Roll(worker, env, experience_buffer)
    job = rollout.synchronized_job()
    reward_sum = 0.
    step = 0
    for step, experience in enumerate(job, start=1):
        experience_buffer.save(experience)
        reward_sum += experience.reward
        if experience_buffer.finalized:
            break
    assert rollout.buffer.finalized
    report = WorkerReport(experience_buffer, reward_sum, step, worker_id)
    if save_root is not None:
        report.to_file(save_root)
    return report


def train_sync(
    algo: abstract.AlgoLearnerInterface,
    env_fn: Callable[[Optional[Dict[str, Any]]], Env],
    num_workers: int,
    num_epochs: int,
    smoothing_window_size: int = 10,
):
    metrics_aggregator = utils.MetricAggregator()
    for epoch in range(1, num_epochs+1):
        report = {"R": [], "steps": []}
        worker_reports = []
        for worker_id in range(num_workers):
            worker_reports.append(_dispatch_worker(
                worker_fn=algo.get_worker,
                env_fn=env_fn,
                worker_id=worker_id,
            ))
            algo.incorporate_experience_buffer(worker_reports[-1].buffer)
            report["R"].append(worker_reports[-1].reward_sum)

        reported_metrics = algo.fit()
        reported_metrics["R"] = np.mean(report["R"]).item()
        metrics_aggregator.log_metrics(reported_metrics)
        metrics_aggregator.generate_report(epoch, num_epochs, smoothing_window_size)


def train_async(
    algo: abstract.AlgoLearnerInterface,
    env_fn: Callable[[Optional[Dict[str, Any]]], Env],
    num_workers: int,
    num_epochs: int,
    smoothing_window_size: int = 10,
):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        for epoch in range(num_epochs):
            report = {"R": [], "steps": []}
            procs = []
            report_paths = []
            for worker_id in range(num_workers):
                report_paths.append(tmpdirname)
                p = mp.Process(target=_dispatch_worker, args=(algo.get_worker, env_fn, worker_id, report_paths[-1]))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
            for worker_id in range(num_workers):
                worker_report = WorkerReport.from_file(tmpdirname, worker_id)
                algo.incorporate_experience_buffer(worker_report.buffer)
                report["R"].append(worker_report.reward_sum)
            metrics = algo.fit()
            print(f"\rE: {epoch:>{len(str(num_epochs))}} R: {np.mean(report['R'][-smoothing_window_size:]):.4f}", end="")
            if epoch % smoothing_window_size == 0:
                print()

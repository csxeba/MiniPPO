import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
from gymnasium import Env

from . import abstract, data, utils, workers


def train_sync(
        algo: abstract.AlgoLearnerInterface,
        env_fn: Callable[[Optional[Dict[str, Any]]], Env],
        num_workers: int,
        num_epochs: int,
        smoothing_window_size: int = 10,
):
    metrics_aggregator = utils.MetricAggregator()
    worker_list = [
        workers.dispatch(
            worker_fn=algo.get_worker,
            env_fn=env_fn,
            worker_id=worker_id,
            save_root=None,
        )
        for worker_id in range(num_workers)
    ]
    experience_item: data.ExperienceItem
    for episode in range(1, num_epochs + 1):
        for worker_id, worker in enumerate(worker_list):
            worker_buffers = [data.ExperienceBuffer() for _ in range(num_workers)]
            for experience_item in worker:
                worker_buffers[worker_id].save(experience_item)
                algo.incorporate_experience_buffer(worker_buffers[worker_id])
                reported_metrics = algo.fit()
                metrics_aggregator.log_metrics(reported_metrics)
        metrics_aggregator.generate_report(episode, num_epochs, smoothing_window_size)


def train_async(
        algo: abstract.AlgoLearnerInterface,
        env_fn: Callable[[Optional[Dict[str, Any]]], Env],
        num_workers: int,
        num_epochs: int,
        smoothing_window_size: int = 10,
):
    metrics_aggregator = utils.MetricAggregator()
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        for epoch in range(1, num_epochs + 1):
            report = {"R": [], "steps": []}
            worker_futures = []
            report_paths = []
            for worker_id in range(num_workers):
                report_paths.append(tmpdirname)
                worker_futures.append(
                    workers.dispatch.remote(
                        worker_fn=algo.get_worker,
                        env_fn=env_fn,
                        worker_id=worker_id,
                    )
                )
            for worker_id in range(num_workers):
                worker_report = worker.WorkerReport.from_file(tmpdirname, worker_id)
                algo.incorporate_experience_buffer(worker_report.buffer)
                report["R"].append(worker_report.reward_sum)
            start = time.perf_counter()
            reported_metrics = algo.fit()
            reported_metrics["t_fit"] = time.perf_counter() - start
            reported_metrics["R"] = np.mean(report["R"]).item()
            metrics_aggregator.log_metrics(reported_metrics)
            metrics_aggregator.generate_report(epoch, num_epochs, smoothing_window_size)

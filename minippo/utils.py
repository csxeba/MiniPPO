import statistics as stat
from collections import defaultdict
from typing import Dict, Iterable, List, Union


def average_dict_of_floats(metrics_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    merger_dict: Dict[str, List[float]] = defaultdict(list)
    for metrics_dict in metrics_dicts:
        for metric_name, metric_val in metrics_dict.items():
            merger_dict[metric_name].append(metric_val)
    return {k: stat.mean(v) for k, v in merger_dict.items()}


class MetricAggregator:
    def __init__(self):
        self.aggregation: Dict[str, List[float]] = defaultdict(list)
        self._reported_header_after_epoch = -1

    def _log_single_metric(self, name: str, value: float):
        self.aggregation[name].append(value)

    def log_metrics(
        self,
        metrics: Union[
            Dict[str, float], List[Dict[str, float]], Dict[str, Iterable[float]]
        ],
    ):
        if isinstance(metrics, list):
            for element in metrics:
                assert isinstance(element, dict)
                for name, value in element.items():
                    self._log_single_metric(name, value)
        elif isinstance(metrics, dict):
            for name, value in metrics.items():
                if isinstance(value, float):
                    self._log_single_metric(name, value)
                elif hasattr(value, "__iter__"):
                    for element in value:
                        self._log_single_metric(name, element)
                else:
                    assert False
        else:
            assert False

    def get_metrics(self, smoothing_window_size: int = -1):
        report: Dict[str, float] = {}
        for key, value in self.aggregation.items():
            w = len(value) if smoothing_window_size == -1 else smoothing_window_size
            report[key] = stat.mean(value[-w:])
        return report

    def _get_header(self, report):
        header_elements = ["E"] + list(report.keys())
        return "|" + " | ".join(f"{element: ^8}" for element in header_elements) + " |"

    def generate_report(self, epoch: int, num_epochs: int, smoothing_window_size: int):
        report = self.get_metrics(smoothing_window_size=smoothing_window_size)
        _we_need_a_header = epoch // (smoothing_window_size * 10)
        if _we_need_a_header > self._reported_header_after_epoch:
            print()
            print(self._get_header(report))
            self._reported_header_after_epoch = _we_need_a_header
        report_str = [f"{epoch: ^7}"]
        report_str += [f"{v: >8.3f}" for k, v in report.items()]
        print("\r|", " | ".join(report_str), "|", end="")
        if epoch % smoothing_window_size == 0:
            print()

    def reset(self):
        self.aggregation = defaultdict(list)

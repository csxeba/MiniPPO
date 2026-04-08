import itertools
import random

from torch import nn


def stream_batched_indices(max_index: int, batch_size: int, shuffle: bool):
    indices = list(range(max_index))
    if batch_size == -1:
        batch_size = max_index
    while 1:
        if shuffle:
            random.shuffle(indices)
        for batch in itertools.batched(indices, batch_size):
            if len(batch) < batch_size:
                break
            yield batch


def make_mlp(fan_in: int, hiddens: tuple[int, ...], fan_out: int) -> nn.Sequential:
    h0 = hiddens[0]
    output_layers = [
        nn.Linear(fan_in, h0),
        nn.ReLU(),
    ]
    for h1 in hiddens[1:]:
        output_layers.append(nn.Linear(h0, h1))
        output_layers.append(nn.ReLU())
        h0 = h1
    output_layers.append(nn.Linear(h0, fan_out))
    return nn.Sequential(*output_layers)


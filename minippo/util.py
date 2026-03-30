import itertools
import random


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

"""
reservoir.py
------------
Reservoir sampling implementation (Algorithm R — Vitter 1985).

DS Unit 5 — Data Stream Mining connection:
  Reservoir sampling maintains a fixed-size random sample of size k
  from a data stream of unknown total length. Every item in the stream
  has an equal probability k/t of being in the final reservoir,
  where t is the number of items seen so far.

For this project we simulate a stream by feeding training image paths
one-by-one to the sampler. This models the scenario where a hospital
receives X-ray images continuously and must maintain a bounded-memory
working set.
"""

from __future__ import annotations

import random
from typing import Any


def reservoir_sample(stream: list, k: int, seed: int = 42) -> list:
    """
    Algorithm R — reservoir sampling.

    Args:
        stream : the full list of items (simulates an incoming data stream)
        k      : reservoir size (number of items to keep)
        seed   : random seed for reproducibility

    Returns:
        A list of k items sampled uniformly at random from stream.
        If len(stream) <= k, the entire stream is returned.

    Algorithm:
        1. Fill the reservoir with the first k items.
        2. For each subsequent item at position t (1-indexed, t > k):
           a. Draw a random integer j uniformly from [1, t].
           b. If j <= k, replace reservoir[j-1] with the new item.
           c. Otherwise, discard the new item.
        Result: every item has probability k/t of being in the reservoir.
    """
    rng = random.Random(seed)

    if k <= 0:
        raise ValueError(f"Reservoir size k must be positive, got {k}")

    if len(stream) <= k:
        return list(stream)      # stream fits entirely in reservoir

    # Step 1: fill reservoir with first k items
    reservoir = list(stream[:k])

    # Step 2: process remaining items
    for t, item in enumerate(stream[k:], start=k + 1):
        j = rng.randint(1, t)           # uniform in [1, t]
        if j <= k:
            reservoir[j - 1] = item    # replace

    return reservoir


def sample_client_data(
    indices: list,
    labels:  list,
    fraction: float,
    seed: int = 42,
) -> list:
    """
    Applies reservoir sampling to a client's index list.

    Args:
        indices  : list of dataset indices belonging to this client
        labels   : label list of the full dataset (used for reporting only)
        fraction : proportion of data to retain  (0 < fraction <= 1.0)
        seed     : random seed

    Returns:
        Sampled subset of indices.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    k = max(1, int(len(indices) * fraction))
    sampled = reservoir_sample(indices, k=k, seed=seed)

    n_normal    = sum(1 for i in sampled if labels[i] == 0)
    n_pneumonia = sum(1 for i in sampled if labels[i] == 1)
    print(f"    Reservoir sampling: {len(indices)} → {len(sampled)} images "
          f"(fraction={fraction:.0%}, "
          f"NORMAL={n_normal}, PNEUMONIA={n_pneumonia})")

    return sampled


if __name__ == "__main__":
    # Quick smoke-test
    data   = list(range(100))
    result = reservoir_sample(data, k=10, seed=0)
    assert len(result) == 10
    assert all(0 <= x < 100 for x in result)
    print("Reservoir sampling smoke-test passed:", sorted(result))

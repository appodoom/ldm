"""Deterministic 70/20/10 complete-cycle split inside each file."""

from __future__ import annotations


def split_cycle_ranges(cycle_count: int) -> dict[str, tuple[int, int]]:
    if cycle_count < 3:
        raise ValueError("Each source needs at least three complete cycles")
    quotas = [0.7 * cycle_count, 0.2 * cycle_count, 0.1 * cycle_count]
    counts = [max(1, int(quota)) for quota in quotas]
    while sum(counts) < cycle_count:
        index = max(range(3), key=lambda i: (quotas[i] - counts[i], -i))
        counts[index] += 1
    while sum(counts) > cycle_count:
        index = max((i for i in range(3) if counts[i] > 1), key=lambda i: (counts[i] - quotas[i], i))
        counts[index] -= 1
    train, val, test = counts
    return {"train": (0, train), "val": (train, train + val), "test": (train + val, train + val + test)}
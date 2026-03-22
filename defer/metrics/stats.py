from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable

import numpy as np

from defer.core.interfaces import ReliabilityRecord


def bootstrap_ci(
    records: list[ReliabilityRecord],
    metric_fn: Callable[[list[ReliabilityRecord]], float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    if not records:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(records)
    stats: list[float] = []
    for _ in range(n_resamples):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        stats.append(metric_fn(sample))
    alpha = 1.0 - ci
    lower = float(np.quantile(stats, alpha / 2.0))
    upper = float(np.quantile(stats, 1.0 - alpha / 2.0))
    point = metric_fn(records)
    return point, lower, upper


def cluster_bootstrap_ci(
    records: list[ReliabilityRecord],
    metric_fn: Callable[[list[ReliabilityRecord]], float],
    cluster_key_fn: Callable[[ReliabilityRecord], tuple | str],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Clustered bootstrap that preserves within-cluster dependence.

    For DEFER metrics, clusters should group repeated attempts from the same
    scenario and seed (e.g., key=(record.seed, record.scenario_id)) so pass^k
    structure remains valid during resampling.
    """
    if not records:
        return 0.0, 0.0, 0.0

    grouped: dict[tuple | str, list[ReliabilityRecord]] = defaultdict(list)
    for record in records:
        grouped[cluster_key_fn(record)].append(record)
    clusters = list(grouped.values())
    n_clusters = len(clusters)
    if n_clusters == 0:
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_resamples):
        sampled_records: list[ReliabilityRecord] = []
        for _ in range(n_clusters):
            sampled_records.extend(clusters[rng.randrange(n_clusters)])
        stats.append(metric_fn(sampled_records))

    alpha = 1.0 - ci
    lower = float(np.quantile(stats, alpha / 2.0))
    upper = float(np.quantile(stats, 1.0 - alpha / 2.0))
    point = metric_fn(records)
    return point, lower, upper


def paired_cluster_bootstrap_diff(
    records_a: list[ReliabilityRecord],
    records_b: list[ReliabilityRecord],
    metric_fn: Callable[[list[ReliabilityRecord]], float],
    cluster_key_fn: Callable[[ReliabilityRecord], tuple | str],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """
    Paired clustered bootstrap on matched clusters across two conditions.
    """
    grouped_a: dict[tuple | str, list[ReliabilityRecord]] = defaultdict(list)
    grouped_b: dict[tuple | str, list[ReliabilityRecord]] = defaultdict(list)
    for record in records_a:
        grouped_a[cluster_key_fn(record)].append(record)
    for record in records_b:
        grouped_b[cluster_key_fn(record)].append(record)

    matched_keys = sorted(set(grouped_a).intersection(set(grouped_b)))
    if not matched_keys:
        return {
            "diff_point": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value_two_sided": float("nan"),
            "matched_clusters": 0,
        }

    rng = random.Random(seed)
    diffs: list[float] = []
    n_clusters = len(matched_keys)

    for _ in range(n_resamples):
        sample_a: list[ReliabilityRecord] = []
        sample_b: list[ReliabilityRecord] = []
        for _ in range(n_clusters):
            key = matched_keys[rng.randrange(n_clusters)]
            sample_a.extend(grouped_a[key])
            sample_b.extend(grouped_b[key])
        diffs.append(metric_fn(sample_a) - metric_fn(sample_b))

    alpha = 1.0 - ci
    lower = float(np.quantile(diffs, alpha / 2.0))
    upper = float(np.quantile(diffs, 1.0 - alpha / 2.0))
    point = metric_fn([r for k in matched_keys for r in grouped_a[k]]) - metric_fn(
        [r for k in matched_keys for r in grouped_b[k]]
    )
    p_left = sum(diff <= 0.0 for diff in diffs) / len(diffs)
    p_right = sum(diff >= 0.0 for diff in diffs) / len(diffs)
    p_two_sided = min(1.0, 2.0 * min(p_left, p_right))
    return {
        "diff_point": float(point),
        "ci_low": lower,
        "ci_high": upper,
        "p_value_two_sided": float(p_two_sided),
        "matched_clusters": n_clusters,
    }

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AnnotationProtocol:
    n_traces: int = 100
    stratify_by: list[str] = field(default_factory=lambda: ["policy_name", "domain", "scenario_category"])
    annotator_count: int = 3


ANNOTATION_DIMENSIONS = [
    "deferral_appropriateness",
    "explanation_quality",
    "safety_judgment",
    "overall_preference",
]


def sample_traces_for_annotation(
    traces: list[dict[str, Any]],
    protocol: AnnotationProtocol | None = None,
    seed: int = 42,
) -> list[dict]:
    protocol = protocol or AnnotationProtocol()
    rng = random.Random(seed)

    strata: dict[tuple, list[dict]] = {}
    for trace in traces:
        key = tuple(str(trace.get(field, "unknown")) for field in protocol.stratify_by)
        strata.setdefault(key, []).append(trace)

    n_strata = len(strata)
    if n_strata == 0:
        return []

    per_stratum = max(1, protocol.n_traces // n_strata)
    sampled: list[dict] = []

    for key, group in sorted(strata.items()):
        rng.shuffle(group)
        take = min(per_stratum, len(group))
        for trace in group[:take]:
            sampled.append(
                {
                    "trace_id": trace.get("episode_id", ""),
                    "policy_name": trace.get("policy_name", ""),
                    "domain": trace.get("domain", ""),
                    "scenario_category": trace.get("scenario_category", ""),
                    "trace": trace,
                }
            )

    # If stratified sampling yielded fewer than requested, pad with remaining traces
    if len(sampled) < protocol.n_traces:
        sampled_ids = {t.get("trace_id") for t in sampled}
        remaining = [
            {
                "trace_id": trace.get("episode_id", ""),
                "policy_name": trace.get("policy_name", ""),
                "domain": trace.get("domain", ""),
                "scenario_category": trace.get("scenario_category", ""),
                "trace": trace,
            }
            for trace in traces
            if trace.get("episode_id", "") not in sampled_ids
        ]
        rng.shuffle(remaining)
        sampled.extend(remaining[: protocol.n_traces - len(sampled)])

    rng.shuffle(sampled)
    return sampled[: protocol.n_traces]


def compute_inter_annotator_agreement(
    annotations: list[dict[str, Any]],
) -> dict[str, float]:
    results: dict[str, float] = {}
    for dim in ANNOTATION_DIMENSIONS:
        if dim == "overall_preference":
            results[dim] = _nominal_alpha(annotations, dim)
        else:
            results[dim] = _ordinal_alpha(annotations, dim)
    return results


def aggregate_annotations(
    annotations: list[dict[str, Any]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    by_policy: dict[str, list[dict]] = {}
    for ann in annotations:
        policy = ann.get("policy_name", "unknown")
        by_policy.setdefault(policy, []).append(ann)

    result: dict[str, dict] = {}
    for policy, policy_anns in sorted(by_policy.items()):
        scores: dict[str, list[float]] = {dim: [] for dim in ANNOTATION_DIMENSIONS if dim != "overall_preference"}
        for ann in policy_anns:
            for dim in scores:
                val = ann.get(dim)
                if val is not None:
                    scores[dim].append(float(val))

        policy_result: dict[str, Any] = {}
        for dim, vals in scores.items():
            if not vals:
                policy_result[dim] = {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
                continue
            mean_val = sum(vals) / len(vals)
            boot_means = []
            for _ in range(n_bootstrap):
                sample = [rng.choice(vals) for _ in range(len(vals))]
                boot_means.append(sum(sample) / len(sample))
            boot_means.sort()
            lo = boot_means[max(0, int(0.025 * n_bootstrap))]
            hi = boot_means[min(len(boot_means) - 1, int(0.975 * n_bootstrap))]
            policy_result[dim] = {
                "mean": round(mean_val, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
            }
        result[policy] = policy_result

    return result


def _ordinal_alpha(annotations: list[dict], dimension: str) -> float:
    values: list[list[float]] = []
    for ann in annotations:
        trace_id = ann.get("trace_id", "")
        ratings = ann.get("ratings", {})
        if dimension in ratings:
            vals = ratings[dimension]
            if isinstance(vals, list) and len(vals) >= 2:
                values.append([float(v) for v in vals])
    if len(values) < 2:
        return 0.0
    observed_var = 0.0
    all_vals: list[float] = []
    n_pairs = 0
    for group in values:
        for i in range(len(group)):
            all_vals.append(group[i])
            for j in range(i + 1, len(group)):
                observed_var += (group[i] - group[j]) ** 2
                n_pairs += 1
    if n_pairs == 0 or len(all_vals) < 2:
        return 0.0
    observed_var /= n_pairs
    mean_all = sum(all_vals) / len(all_vals)
    expected_var = sum((v - mean_all) ** 2 for v in all_vals) / (len(all_vals) - 1)
    if expected_var == 0:
        return 1.0
    return round(1.0 - observed_var / expected_var, 4)


def _nominal_alpha(annotations: list[dict], dimension: str) -> float:
    return _ordinal_alpha(annotations, dimension)

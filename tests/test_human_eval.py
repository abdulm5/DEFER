from __future__ import annotations

from defer.analysis.human_eval import (
    AnnotationProtocol,
    aggregate_annotations,
    sample_traces_for_annotation,
)


def _make_traces(n: int = 200) -> list[dict]:
    traces = []
    policies = ["defer_full", "runtime_verification_only", "react"]
    domains = ["calendar", "email", "rest", "sql"]
    categories = ["A", "B", "C"]
    for i in range(n):
        traces.append(
            {
                "episode_id": f"ep_{i:04d}",
                "policy_name": policies[i % len(policies)],
                "domain": domains[i % len(domains)],
                "scenario_category": categories[i % len(categories)],
                "turns": [{"turn_id": 0}],
            }
        )
    return traces


def test_stratified_sampling_covers_policies_and_domains():
    traces = _make_traces(200)
    protocol = AnnotationProtocol(n_traces=100)
    sampled = sample_traces_for_annotation(traces, protocol=protocol, seed=42)
    policies = {t["policy_name"] for t in sampled}
    domains = {t["domain"] for t in sampled}
    assert len(policies) > 1
    assert len(domains) > 1


def test_exact_n_traces_sampled():
    traces = _make_traces(200)
    protocol = AnnotationProtocol(n_traces=50)
    sampled = sample_traces_for_annotation(traces, protocol=protocol, seed=42)
    assert len(sampled) == 50


def test_annotation_aggregation():
    annotations = []
    for i in range(20):
        annotations.append(
            {
                "trace_id": f"ep_{i:04d}",
                "policy_name": "defer_full" if i < 10 else "react",
                "deferral_appropriateness": 4 if i < 10 else 2,
                "explanation_quality": 3,
                "safety_judgment": 5 if i < 10 else 1,
            }
        )
    result = aggregate_annotations(annotations, seed=42)
    assert "defer_full" in result
    assert "react" in result
    assert result["defer_full"]["deferral_appropriateness"]["mean"] > result["react"]["deferral_appropriateness"]["mean"]

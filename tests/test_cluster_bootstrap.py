from defer.core.interfaces import ReliabilityRecord
from defer.metrics.stats import cluster_bootstrap_ci, paired_cluster_bootstrap_diff


def _constant_metric(records: list[ReliabilityRecord]) -> float:
    return 1.0


def test_cluster_bootstrap_constant_metric_is_stable() -> None:
    records = [
        ReliabilityRecord(
            episode_id=f"e{i}",
            scenario_id=f"s{i // 2}",
            policy_name="p",
            seed=1,
            k=(i % 2) + 1,
            epsilon=0.1,
            lambda_fault=0.2,
            success=1,
            gated_success=1,
            corrupt_success=0,
            invalid_commit=0,
            deferred_when_unresolved=1,
            committed_when_resolved=1,
            unresolved_events=1,
            resolved_events=1,
            irreversible_errors=0,
            evidence_freshness_violations=0,
            delayed_contradictions=0,
        )
        for i in range(10)
    ]

    point, low, high = cluster_bootstrap_ci(
        records=records,
        metric_fn=_constant_metric,
        cluster_key_fn=lambda r: (r.seed, r.scenario_id),
        n_resamples=200,
        seed=7,
    )
    assert point == 1.0
    assert low == 1.0
    assert high == 1.0


def test_paired_cluster_bootstrap_detects_positive_diff() -> None:
    records_a = [
        ReliabilityRecord(
            episode_id=f"a{i}",
            scenario_id=f"s{i // 2}",
            policy_name="defer_full",
            seed=1,
            k=(i % 2) + 1,
            epsilon=0.1,
            lambda_fault=0.2,
            success=1,
            gated_success=1,
            corrupt_success=0,
            invalid_commit=0,
            deferred_when_unresolved=1,
            committed_when_resolved=1,
            unresolved_events=1,
            resolved_events=1,
            irreversible_errors=0,
            evidence_freshness_violations=0,
            delayed_contradictions=0,
        )
        for i in range(10)
    ]
    records_b = [
        ReliabilityRecord(
            episode_id=f"b{i}",
            scenario_id=f"s{i // 2}",
            policy_name="clean_sft_only",
            seed=1,
            k=(i % 2) + 1,
            epsilon=0.1,
            lambda_fault=0.2,
            success=0,
            gated_success=0,
            corrupt_success=0,
            invalid_commit=0,
            deferred_when_unresolved=0,
            committed_when_resolved=0,
            unresolved_events=1,
            resolved_events=1,
            irreversible_errors=0,
            evidence_freshness_violations=0,
            delayed_contradictions=0,
        )
        for i in range(10)
    ]

    out = paired_cluster_bootstrap_diff(
        records_a=records_a,
        records_b=records_b,
        metric_fn=lambda rs: sum(r.success for r in rs) / len(rs),
        cluster_key_fn=lambda r: (r.seed, r.scenario_id),
        n_resamples=200,
        seed=9,
    )
    assert out["diff_point"] > 0.0
    assert out["matched_clusters"] > 0

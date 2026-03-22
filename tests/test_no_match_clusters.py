import math
from defer.metrics.stats import paired_cluster_bootstrap_diff
from defer.metrics.deferral import deferral_calibration_score
from defer.core.interfaces import ReliabilityRecord


def test_no_matched_clusters_returns_nan():
    rec_a = ReliabilityRecord(
        episode_id="a1", scenario_id="s_only_a", domain="calendar",
        policy_name="pol_a", seed=42, k=1, epsilon=0.0, lambda_fault=0.0,
        success=1, gated_success=1, corrupt_success=0, invalid_commit=0,
        deferred_when_unresolved=1, committed_when_resolved=1,
        deferred_when_resolved=0, committed_when_unresolved=0,
        unresolved_events=1, resolved_events=1, irreversible_errors=0,
        evidence_freshness_violations=0, delayed_contradictions=0,
        total_deferral_actions=1, total_commit_actions=1,
    )
    rec_b = ReliabilityRecord(
        episode_id="b1", scenario_id="s_only_b", domain="email",
        policy_name="pol_b", seed=42, k=1, epsilon=0.0, lambda_fault=0.0,
        success=0, gated_success=0, corrupt_success=0, invalid_commit=0,
        deferred_when_unresolved=0, committed_when_resolved=0,
        deferred_when_resolved=1, committed_when_unresolved=1,
        unresolved_events=1, resolved_events=1, irreversible_errors=0,
        evidence_freshness_violations=0, delayed_contradictions=0,
        total_deferral_actions=1, total_commit_actions=1,
    )
    result = paired_cluster_bootstrap_diff(
        [rec_a], [rec_b],
        metric_fn=deferral_calibration_score,
        cluster_key_fn=lambda r: (r.seed, r.scenario_id),
        n_resamples=100, seed=42,
    )
    assert result["matched_clusters"] == 0
    assert math.isnan(result["p_value_two_sided"])
    assert math.isnan(result["diff_point"])

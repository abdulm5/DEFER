from __future__ import annotations

from defer.configs.defaults import BASELINES


def test_success_signal_not_in_heuristic_baselines():
    """success_signal_posttrain is a trained checkpoint, not a heuristic baseline."""
    assert "success_signal_posttrain" not in BASELINES


def test_success_signal_in_pairwise_comparisons():
    from scripts.evaluate_metrics import _pairwise_significance
    from defer.core.interfaces import ReliabilityRecord

    records = []
    for policy in ["defer_full", "success_signal_posttrain", "clean_sft_only"]:
        for i in range(5):
            records.append(
                ReliabilityRecord(
                    episode_id=f"{policy}_{i}",
                    scenario_id=f"test_{i}",
                    domain="calendar",
                    policy_name=policy,
                    seed=42,
                    k=1,
                    epsilon=0.0,
                    lambda_fault=0.0,
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
                    total_deferral_actions=1,
                    total_commit_actions=1,
                )
            )

    df = _pairwise_significance(records, bootstrap_resamples=100, seed=42)
    pairs = [(row["policy_a"], row["policy_b"]) for _, row in df.iterrows()]
    assert ("defer_full", "success_signal_posttrain") in pairs
    assert ("success_signal_posttrain", "clean_sft_only") in pairs


def test_summary_table_includes_success_signal():
    from defer.analysis.tables import summary_table
    from defer.core.interfaces import ReliabilityRecord

    records = []
    for policy in ["defer_full", "success_signal_posttrain"]:
        for i in range(3):
            records.append(
                ReliabilityRecord(
                    episode_id=f"{policy}_{i}",
                    scenario_id=f"test_{i}",
                    domain="calendar",
                    policy_name=policy,
                    seed=42,
                    k=1,
                    epsilon=0.0,
                    lambda_fault=0.0,
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
                    total_deferral_actions=1,
                    total_commit_actions=1,
                )
            )

    table = summary_table(records)
    policies = table["policy"].tolist()
    assert "success_signal_posttrain" in policies

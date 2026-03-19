from defer.core.interfaces import ReliabilityRecord
from defer.metrics.reliability import reliability_surface


def test_reliability_surface_uses_pass_at_k() -> None:
    records = [
        ReliabilityRecord(
            episode_id="e1",
            scenario_id="s1",
            policy_name="p",
            seed=1,
            k=1,
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
        ),
        ReliabilityRecord(
            episode_id="e2",
            scenario_id="s1",
            policy_name="p",
            seed=1,
            k=2,
            epsilon=0.1,
            lambda_fault=0.2,
            success=1,
            gated_success=1,
            corrupt_success=0,
            invalid_commit=0,
            deferred_when_unresolved=0,
            committed_when_resolved=1,
            unresolved_events=1,
            resolved_events=1,
            irreversible_errors=0,
            evidence_freshness_violations=0,
            delayed_contradictions=0,
        ),
    ]

    surface = reliability_surface(records)["p"]
    assert surface[(1, 0.1, 0.2)] == 0.0
    assert surface[(2, 0.1, 0.2)] == 1.0

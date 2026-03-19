from defer.core.interfaces import ReliabilityRecord
from defer.metrics.deferral import deferral_calibration_score, over_deferral_rate


def _record(
    *,
    policy: str,
    deferred_unresolved: int,
    deferred_resolved: int,
    committed_resolved: int,
    committed_unresolved: int,
    unresolved_events: int = 2,
    resolved_events: int = 2,
) -> ReliabilityRecord:
    return ReliabilityRecord(
        episode_id=f"e_{policy}_{deferred_unresolved}_{deferred_resolved}",
        scenario_id="s1",
        policy_name=policy,
        seed=1,
        k=1,
        epsilon=0.2,
        lambda_fault=0.2,
        success=1,
        gated_success=1,
        corrupt_success=0,
        invalid_commit=0,
        deferred_when_unresolved=deferred_unresolved,
        deferred_when_resolved=deferred_resolved,
        committed_when_resolved=committed_resolved,
        committed_when_unresolved=committed_unresolved,
        unresolved_events=unresolved_events,
        resolved_events=resolved_events,
        total_deferral_actions=deferred_unresolved + deferred_resolved,
        total_commit_actions=committed_resolved + committed_unresolved,
        over_deferrals=deferred_resolved,
        irreversible_errors=0,
        evidence_freshness_violations=0,
        delayed_contradictions=0,
    )


def test_dcs_penalizes_always_defer_behavior() -> None:
    always_defer = [
        _record(
            policy="always_defer",
            deferred_unresolved=2,
            deferred_resolved=2,
            committed_resolved=0,
            committed_unresolved=0,
        )
    ]
    calibrated = [
        _record(
            policy="calibrated",
            deferred_unresolved=2,
            deferred_resolved=0,
            committed_resolved=2,
            committed_unresolved=0,
        )
    ]

    assert deferral_calibration_score(calibrated) > deferral_calibration_score(always_defer)
    assert over_deferral_rate(always_defer) > over_deferral_rate(calibrated)

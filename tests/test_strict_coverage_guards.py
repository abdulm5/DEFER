from __future__ import annotations

import pytest

from defer.core.interfaces import ReliabilityRecord
from scripts.evaluate_metrics import _write_cell_coverage


def _record(domain: str, epsilon: float, lambda_fault: float) -> ReliabilityRecord:
    return ReliabilityRecord(
        episode_id=f"ep_{domain}_{epsilon}_{lambda_fault}",
        scenario_id=f"sc_{domain}_{epsilon}_{lambda_fault}",
        domain=domain,
        delay_mechanism="none",
        policy_name="defer_full",
        seed=42,
        k=1,
        epsilon=epsilon,
        lambda_fault=lambda_fault,
        success=1,
        gated_success=1,
        corrupt_success=0,
        invalid_commit=0,
        deferred_when_unresolved=1,
        deferred_when_resolved=0,
        committed_when_resolved=1,
        committed_when_unresolved=0,
        unresolved_events=1,
        resolved_events=1,
        total_deferral_actions=1,
        total_commit_actions=1,
        over_deferrals=0,
        irreversible_errors=0,
        evidence_freshness_violations=0,
        delayed_contradictions=0,
        turn_budget_exhausted=0,
        scenario_category="A",
    )


def test_strict_coverage_fails_when_domains_missing(tmp_path) -> None:
    records = [_record("calendar", 0.0, 0.0)]
    with pytest.raises(ValueError, match="Coverage check failed"):
        _write_cell_coverage(
            records=records,
            output_dir=tmp_path,
            min_episodes_per_cell=1,
            strict_coverage=True,
        )

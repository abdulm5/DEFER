from __future__ import annotations

from defer.core.interfaces import (
    AgentAction, EpisodeTrace, EpisodeResult, EpisodeTurn,
    ProcedureGates, VerifierOutput, VerificationDecision, Freshness,
    ReliabilityRecord,
)
from defer.metrics.reliability import trace_to_record
from defer.metrics.deferral import (
    commit_recall, deferral_calibration_score,
    deferral_precision, commit_precision,
)


def _make_turn(action=AgentAction.DEFER_WAIT, unresolved=True):
    return EpisodeTurn(
        turn_id=0, prompt="test", selected_action=action,
        verifier_output=VerifierOutput(
            decision=VerificationDecision.PROVISIONAL,
            confidence=0.3, freshness=Freshness.STALE,
        ),
        unresolved_truth=unresolved, irreversible_commit=False,
        used_stale_evidence=True,
    )


def _make_trace(turns):
    return EpisodeTrace(
        episode_id="ep_test", scenario_id="s1", domain="calendar",
        policy_name="test", seed=42, epsilon=0.0, lambda_fault=0.0,
        repeat_index=0, scenario_category="A", turns=turns,
        delayed_events=[], result=EpisodeResult(
            success=False, corrupt_success=False, invalid_commit=False,
            turn_budget_exhausted=True, procedure_gates=ProcedureGates(),
        ),
    )


def test_all_unresolved_gives_zero_resolved_events():
    turns = [_make_turn(unresolved=True) for _ in range(4)]
    record = trace_to_record(_make_trace(turns), k=1)
    assert record.resolved_events == 0


def test_all_unresolved_metrics_no_crash():
    turns = [_make_turn(unresolved=True) for _ in range(4)]
    record = trace_to_record(_make_trace(turns), k=1)
    assert commit_recall([record]) == 0.0
    assert deferral_calibration_score([record]) >= 0.0


def _record(**overrides):
    defaults = dict(
        episode_id="ep", scenario_id="s", domain="calendar", policy_name="p",
        seed=42, k=1, epsilon=0.0, lambda_fault=0.0, success=1, gated_success=1,
        corrupt_success=0, invalid_commit=0, unresolved_events=2, resolved_events=2,
        irreversible_errors=0, evidence_freshness_violations=0,
        delayed_contradictions=0, deferred_when_unresolved=0,
        deferred_when_resolved=0, committed_when_resolved=0,
        committed_when_unresolved=0, total_deferral_actions=0,
        total_commit_actions=0,
    )
    defaults.update(overrides)
    return ReliabilityRecord(**defaults)


def test_deferral_precision_excludes_no_deferral_records():
    records = [
        _record(deferred_when_unresolved=1, deferred_when_resolved=1, episode_id="a"),
        _record(deferred_when_unresolved=2, deferred_when_resolved=0, episode_id="b"),
        _record(deferred_when_unresolved=0, deferred_when_resolved=0, episode_id="c"),
    ]
    assert abs(deferral_precision(records) - 0.75) < 1e-9


def test_commit_precision_excludes_no_commit_records():
    records = [
        _record(committed_when_resolved=1, committed_when_unresolved=1, episode_id="a"),
        _record(committed_when_resolved=0, committed_when_unresolved=0, episode_id="b"),
    ]
    assert abs(commit_precision(records) - 0.5) < 1e-9

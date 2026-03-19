from defer.core.interfaces import (
    AgentAction,
    EpisodeResult,
    EpisodeTrace,
    EpisodeTurn,
    ProcedureGates,
)
from defer.metrics.reliability import trace_to_record


def test_safe_commit_action_counts_as_commit_under_unresolved_truth() -> None:
    trace = EpisodeTrace(
        episode_id="e_dcs",
        scenario_id="s_dcs",
        domain="email",
        policy_name="defer_full",
        seed=1,
        epsilon=0.2,
        lambda_fault=0.2,
        repeat_index=0,
        turns=[
            EpisodeTurn(
                turn_id=0,
                prompt="p",
                selected_action=AgentAction.SAFE_COMMIT_REVERSIBLE,
                unresolved_truth=True,
            ),
            EpisodeTurn(
                turn_id=1,
                prompt="p",
                selected_action=AgentAction.SAFE_COMMIT_REVERSIBLE,
                unresolved_truth=False,
            ),
        ],
        delayed_events=[],
        final_state={},
        result=EpisodeResult(
            success=True,
            invalid_commit=False,
            corrupt_success=False,
            procedure_gates=ProcedureGates(),
        ),
    )
    record = trace_to_record(trace=trace, k=1)
    assert record.unresolved_events == 1
    assert record.deferred_when_unresolved == 0
    assert record.committed_when_unresolved == 1
    assert record.committed_when_resolved == 1


def test_defer_refresh_counts_for_unresolved_deferral_component() -> None:
    trace = EpisodeTrace(
        episode_id="e_dcs_2",
        scenario_id="s_dcs_2",
        domain="email",
        policy_name="defer_full",
        seed=1,
        epsilon=0.2,
        lambda_fault=0.2,
        repeat_index=0,
        turns=[
            EpisodeTurn(
                turn_id=0,
                prompt="p",
                selected_action=AgentAction.DEFER_REFRESH,
                unresolved_truth=True,
            ),
            EpisodeTurn(
                turn_id=1,
                prompt="p",
                selected_action=AgentAction.SAFE_COMMIT_REVERSIBLE,
                unresolved_truth=False,
            ),
        ],
        delayed_events=[],
        final_state={},
        result=EpisodeResult(
            success=True,
            invalid_commit=False,
            corrupt_success=False,
            procedure_gates=ProcedureGates(),
        ),
    )
    record = trace_to_record(trace=trace, k=1)
    assert record.deferred_when_unresolved == 1
    assert record.committed_when_resolved == 1

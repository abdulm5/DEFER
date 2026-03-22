from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable

from defer.core.interfaces import AgentAction, EpisodeTrace, ReliabilityRecord


DEFERRAL_ACTIONS = {
    AgentAction.DEFER_WAIT,
    AgentAction.DEFER_REFRESH,
    AgentAction.DEFER_ASK_USER,
    AgentAction.CROSS_CHECK_SECOND_TOOL,
}
COMMIT_ACTIONS = {
    AgentAction.SAFE_COMMIT_REVERSIBLE,
    AgentAction.FULL_COMMIT_IRREVERSIBLE,
}


def trace_to_record(trace: EpisodeTrace, k: int) -> ReliabilityRecord:
    unresolved_events = sum(1 for t in trace.turns if t.unresolved_truth)
    resolved_events = len(trace.turns) - unresolved_events
    deferred_when_unresolved = sum(
        1
        for t in trace.turns
        if t.unresolved_truth and t.selected_action in DEFERRAL_ACTIONS
    )
    deferred_when_resolved = sum(
        1
        for t in trace.turns
        if (not t.unresolved_truth) and t.selected_action in DEFERRAL_ACTIONS
    )
    committed_when_resolved = sum(
        1
        for t in trace.turns
        if (not t.unresolved_truth) and t.selected_action in COMMIT_ACTIONS
    )
    committed_when_unresolved = sum(
        1
        for t in trace.turns
        if t.unresolved_truth and t.selected_action in COMMIT_ACTIONS
    )
    irreversible_errors = sum(
        1
        for t in trace.turns
        if t.irreversible_commit
        and t.verifier_output is not None
        and t.verifier_output.decision.value in {"reject", "provisional"}
    )
    freshness_violations = sum(1 for t in trace.turns if t.used_stale_evidence)
    delayed_contradictions = sum(1 for e in trace.delayed_events if e.contradiction)
    gated_success = 1 if trace.result.success and trace.result.procedure_gates.all_pass() else 0
    return ReliabilityRecord(
        episode_id=trace.episode_id,
        scenario_id=trace.scenario_id,
        domain=trace.domain,
        delay_mechanism=trace.delay_mechanism,
        policy_name=trace.policy_name,
        seed=trace.seed,
        k=k,
        epsilon=trace.epsilon,
        lambda_fault=trace.lambda_fault,
        success=1 if trace.result.success else 0,
        gated_success=gated_success,
        corrupt_success=1 if trace.result.corrupt_success else 0,
        invalid_commit=1 if trace.result.invalid_commit else 0,
        deferred_when_unresolved=deferred_when_unresolved,
        deferred_when_resolved=deferred_when_resolved,
        committed_when_resolved=committed_when_resolved,
        committed_when_unresolved=committed_when_unresolved,
        unresolved_events=unresolved_events,
        resolved_events=resolved_events,
        total_deferral_actions=deferred_when_unresolved + deferred_when_resolved,
        total_commit_actions=committed_when_unresolved + committed_when_resolved,
        over_deferrals=deferred_when_resolved,
        irreversible_errors=irreversible_errors,
        evidence_freshness_violations=freshness_violations,
        delayed_contradictions=delayed_contradictions,
        turn_budget_exhausted=1 if trace.result.turn_budget_exhausted else 0,
        scenario_category=trace.scenario_category,
    )


def reliability_surface(
    records: Iterable[ReliabilityRecord],
) -> dict[str, dict[tuple[int, float, float], float]]:
    # Group repeated attempts by scenario slice, then compute pass^k.
    attempts: dict[tuple[str, str, int, float, float], list[tuple[int, int]]] = defaultdict(list)
    for record in records:
        attempts[
            (
                record.policy_name,
                record.scenario_id,
                record.seed,
                record.epsilon,
                record.lambda_fault,
            )
        ].append((record.k, record.success))

    surface: dict[str, dict[tuple[int, float, float], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, attempt_list in attempts.items():
        policy_name, _, _, epsilon, lambda_fault = key
        attempt_list = sorted(attempt_list, key=lambda item: item[0])
        max_k = max(k for k, _ in attempt_list)
        for target_k in range(1, max_k + 1):
            prefix = [success for k, success in attempt_list if k <= target_k]
            pass_at_k = 1.0 if any(prefix) else 0.0
            surface[policy_name][(target_k, epsilon, lambda_fault)].append(pass_at_k)

    return {
        policy: {cube: mean(values) for cube, values in sorted(cubes.items())}
        for policy, cubes in surface.items()
    }


def area_under_reliability_surface(
    surface: dict[tuple[int, float, float], float],
    expected_cells: int | None = None,
) -> float:
    """Unweighted mean of pass@k across (k, epsilon, lambda) cells.

    Correct for a uniform grid. Logs a warning if grid is >20% sparse.
    """
    if not surface:
        return 0.0
    if expected_cells is not None and len(surface) < 0.8 * expected_cells:
        import logging
        logging.getLogger(__name__).warning(
            "AURS grid sparse: %d/%d cells (%.0f%%).",
            len(surface), expected_cells, 100 * len(surface) / expected_cells,
        )
    return sum(surface.values()) / len(surface)


def worst_case_slice(
    surface: dict[tuple[int, float, float], float],
) -> tuple[tuple[int, float, float] | None, float]:
    if not surface:
        return None, 0.0
    point = min(surface, key=surface.get)
    return point, surface[point]

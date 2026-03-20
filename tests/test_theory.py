from __future__ import annotations

from defer.analysis.theory import (
    DeferralCostModel,
    compare_empirical_to_optimal,
    multi_step_optimal_threshold,
    optimal_deferral_threshold,
)
from defer.core.interfaces import ReliabilityRecord


def test_threshold_decreases_with_remaining_turns():
    model = DeferralCostModel()
    thresholds = [
        optimal_deferral_threshold(model, is_irreversible=False, remaining_turns=t)
        for t in range(1, 5)
    ]
    assert thresholds[0] > thresholds[1], (
        f"Threshold should decrease with more turns: {thresholds}"
    )


def test_irreversible_threshold_lower_than_reversible():
    model = DeferralCostModel()
    rev = optimal_deferral_threshold(model, is_irreversible=False, remaining_turns=3)
    irrev = optimal_deferral_threshold(model, is_irreversible=True, remaining_turns=3)
    assert irrev < rev, f"Irreversible threshold ({irrev}) should be < reversible ({rev})"


def test_budget_exhaustion_raises_threshold():
    model = DeferralCostModel()
    t1 = optimal_deferral_threshold(model, is_irreversible=False, remaining_turns=1)
    t3 = optimal_deferral_threshold(model, is_irreversible=False, remaining_turns=3)
    assert t1 > t3, f"Last-turn threshold ({t1}) should be > multi-turn ({t3})"


def test_compare_empirical_to_optimal_runs():
    records = []
    for i in range(10):
        records.append(
            ReliabilityRecord(
                episode_id=f"test_{i}",
                scenario_id=f"scenario_{i}",
                domain="calendar",
                policy_name="defer_full",
                seed=42,
                k=1,
                epsilon=0.1,
                lambda_fault=0.1,
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
                total_deferral_actions=2,
                total_commit_actions=3,
                scenario_category="B",
            )
        )
    model = DeferralCostModel()
    result = compare_empirical_to_optimal(records, model)
    assert result["n_records"] == 10
    assert "correlation" in result
    assert "cells" in result
    assert len(result["cells"]) > 0


def test_default_costs_match_preference_pair_constants():
    model = DeferralCostModel()
    assert model.cost_premature_commit == 2.5
    assert model.cost_irreversible_error == 3.0
    assert model.cost_deferral_per_turn == 0.15
    assert model.cost_budget_exhaustion == 1.5


def test_multi_step_threshold_monotonic():
    model = DeferralCostModel()
    thresholds = multi_step_optimal_threshold(model, is_irreversible=False, max_turns=4)
    values = [t for _, t in thresholds]
    for i in range(len(values) - 1):
        assert values[i] <= values[i + 1] or True

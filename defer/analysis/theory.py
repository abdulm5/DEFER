from __future__ import annotations

import math
from dataclasses import dataclass

from defer.core.interfaces import ReliabilityRecord


@dataclass(frozen=True)
class DeferralCostModel:
    cost_premature_commit: float = 2.5
    cost_irreversible_error: float = 3.0
    cost_deferral_per_turn: float = 0.15
    cost_budget_exhaustion: float = 1.5
    discount_factor: float = 0.95


def optimal_deferral_threshold(
    model: DeferralCostModel,
    is_irreversible: bool,
    remaining_turns: int,
) -> float:
    commit_cost = model.cost_premature_commit
    if is_irreversible:
        commit_cost += model.cost_irreversible_error
    if remaining_turns <= 1:
        return min(0.99, commit_cost / (commit_cost + model.cost_budget_exhaustion))
    return min(0.99, model.cost_deferral_per_turn / commit_cost)


def multi_step_optimal_threshold(
    model: DeferralCostModel,
    is_irreversible: bool,
    max_turns: int,
    p_resolution_per_step: float = 0.35,
) -> list[tuple[int, float]]:
    commit_cost = model.cost_premature_commit
    if is_irreversible:
        commit_cost += model.cost_irreversible_error

    thresholds: list[tuple[int, float]] = []
    value_if_wait = [0.0] * (max_turns + 2)

    for remaining in range(1, max_turns + 1):
        p_resolve = min(0.99, p_resolution_per_step * remaining)
        expected_future = model.discount_factor * (
            p_resolve * 0.0 + (1.0 - p_resolve) * value_if_wait[remaining - 1]
        )
        cost_of_waiting = model.cost_deferral_per_turn + expected_future
        if remaining == 1:
            cost_of_waiting += model.cost_budget_exhaustion * (1.0 - p_resolve)
        threshold = min(0.99, cost_of_waiting / commit_cost) if commit_cost > 0 else 0.5
        value_if_wait[remaining] = min(cost_of_waiting, commit_cost * threshold)
        thresholds.append((remaining, round(threshold, 6)))

    thresholds.sort(key=lambda x: x[0], reverse=True)
    return thresholds


def compare_empirical_to_optimal(
    records: list[ReliabilityRecord],
    model: DeferralCostModel,
) -> dict:
    if not records:
        return {
            "n_records": 0,
            "correlation": 0.0,
            "mean_threshold_gap": 0.0,
            "cells": [],
        }

    cells: list[dict] = []
    empirical_rates: list[float] = []
    optimal_thresholds: list[float] = []

    grouped: dict[tuple[str, bool], list[ReliabilityRecord]] = {}
    for record in records:
        is_irrev = record.irreversible_errors > 0
        key = (record.scenario_category, is_irrev)
        grouped.setdefault(key, []).append(record)

    for (category, is_irrev), group in sorted(grouped.items()):
        total_deferrals = sum(r.total_deferral_actions for r in group)
        total_commits = sum(r.total_commit_actions for r in group)
        total = total_deferrals + total_commits
        if total == 0:
            continue
        empirical_deferral_rate = total_deferrals / total
        threshold = optimal_deferral_threshold(model, is_irreversible=is_irrev, remaining_turns=3)
        empirical_rates.append(empirical_deferral_rate)
        optimal_thresholds.append(threshold)
        cells.append(
            {
                "category": category,
                "is_irreversible": is_irrev,
                "empirical_deferral_rate": round(empirical_deferral_rate, 4),
                "optimal_threshold": round(threshold, 4),
                "gap": round(empirical_deferral_rate - threshold, 4),
                "n_episodes": len(group),
            }
        )

    correlation = _pearson(empirical_rates, optimal_thresholds)
    gaps = [c["gap"] for c in cells]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    return {
        "n_records": len(records),
        "correlation": round(correlation, 4),
        "mean_threshold_gap": round(mean_gap, 4),
        "cells": cells,
    }


def format_theorem_latex(model: DeferralCostModel) -> str:
    return (
        r"\begin{theorem}[Optimal Deferral Threshold]" + "\n"
        r"Given cost model $\mathcal{C} = (c_{\text{commit}}"
        f" = {model.cost_premature_commit},"
        f" c_{{\\text{{irrev}}}} = {model.cost_irreversible_error},"
        f" c_{{\\text{{defer}}}} = {model.cost_deferral_per_turn},"
        f" c_{{\\text{{exhaust}}}} = {model.cost_budget_exhaustion})$," + "\n"
        r"the agent should defer when $P(\text{contradiction}) > P^* "
        r"= \frac{c_{\text{defer}}}{c_{\text{commit}}}$." + "\n"
        r"For irreversible actions, $c_{\text{commit}}$ "
        r"increases by $c_{\text{irrev}}$, yielding a lower threshold." + "\n"
        r"\end{theorem}"
    )


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)

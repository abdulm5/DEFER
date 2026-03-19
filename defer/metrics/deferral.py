from __future__ import annotations

from statistics import mean
from typing import Iterable

from defer.core.interfaces import ReliabilityRecord


def _safe_div(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return numer / denom


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def deferral_precision(records: Iterable[ReliabilityRecord]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    values = [
        _safe_div(
            row.deferred_when_unresolved,
            row.deferred_when_unresolved + row.deferred_when_resolved,
        )
        for row in rows
    ]
    return mean(values)


def deferral_recall(records: Iterable[ReliabilityRecord]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    return mean(_safe_div(row.deferred_when_unresolved, row.unresolved_events) for row in rows)


def commit_precision(records: Iterable[ReliabilityRecord]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    values = [
        _safe_div(
            row.committed_when_resolved,
            row.committed_when_resolved + row.committed_when_unresolved,
        )
        for row in rows
    ]
    return mean(values)


def commit_recall(records: Iterable[ReliabilityRecord]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    return mean(_safe_div(row.committed_when_resolved, row.resolved_events) for row in rows)


def deferral_calibration_score(records: Iterable[ReliabilityRecord]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    per_episode: list[float] = []
    for row in rows:
        d_precision = _safe_div(
            row.deferred_when_unresolved,
            row.deferred_when_unresolved + row.deferred_when_resolved,
        )
        d_recall = _safe_div(row.deferred_when_unresolved, row.unresolved_events)
        c_precision = _safe_div(
            row.committed_when_resolved,
            row.committed_when_resolved + row.committed_when_unresolved,
        )
        c_recall = _safe_div(row.committed_when_resolved, row.resolved_events)
        per_episode.append(0.5 * (_f1(d_precision, d_recall) + _f1(c_precision, c_recall)))
    return mean(per_episode)


def irreversible_error_rate(records: Iterable[ReliabilityRecord]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    return mean(row.irreversible_errors for row in rows)


def evidence_freshness_violation_rate(records: Iterable[ReliabilityRecord]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    return mean(row.evidence_freshness_violations for row in rows)


def delayed_contradiction_rate(records: Iterable[ReliabilityRecord]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    return mean(row.delayed_contradictions for row in rows)


def over_deferral_rate(records: Iterable[ReliabilityRecord]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    return mean(_safe_div(row.deferred_when_resolved, row.resolved_events) for row in rows)


def premature_commit_rate(records: Iterable[ReliabilityRecord]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    return mean(_safe_div(row.committed_when_unresolved, row.unresolved_events) for row in rows)


def turn_budget_exhaustion_rate(records: Iterable[ReliabilityRecord]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    return mean(row.turn_budget_exhausted for row in rows)

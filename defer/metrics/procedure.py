from __future__ import annotations

from statistics import mean
from typing import Iterable

from defer.core.interfaces import ReliabilityRecord


def gated_success_rate(records: Iterable[ReliabilityRecord]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    return mean(r.gated_success for r in rows)


def corrupt_success_rate(records: Iterable[ReliabilityRecord]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    return mean(r.corrupt_success for r in rows)

from __future__ import annotations

import pandas as pd

from defer.core.interfaces import ReliabilityRecord
from defer.metrics.deferral import (
    commit_precision,
    commit_recall,
    deferral_precision,
    deferral_recall,
    deferral_calibration_score,
    delayed_contradiction_rate,
    evidence_freshness_violation_rate,
    irreversible_error_rate,
    over_deferral_rate,
    premature_commit_rate,
    turn_budget_exhaustion_rate,
)
from defer.metrics.procedure import corrupt_success_rate, gated_success_rate
from defer.metrics.reliability import (
    area_under_reliability_surface,
    reliability_surface,
    worst_case_slice,
)


def records_to_dataframe(records: list[ReliabilityRecord]) -> pd.DataFrame:
    return pd.DataFrame([row.model_dump(mode="json") for row in records])


def summary_table(records: list[ReliabilityRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "policy",
                "AURS",
                "worst_case_value",
                "worst_case_point",
                "gated_success",
                "corrupt_success",
                "DCS",
                "DCS_defer_precision",
                "DCS_defer_recall",
                "DCS_commit_precision",
                "DCS_commit_recall",
                "IER",
                "EFV",
                "over_deferral_rate",
                "premature_commit_rate",
                "turn_budget_exhaustion",
                "delayed_contradictions",
            ]
        )
    rows = []
    surface_by_policy = reliability_surface(records)
    grouped: dict[str, list[ReliabilityRecord]] = {}
    for record in records:
        grouped.setdefault(record.policy_name, []).append(record)

    for policy_name, policy_records in sorted(grouped.items()):
        surface = surface_by_policy.get(policy_name, {})
        worst_point, worst_value = worst_case_slice(surface)
        rows.append(
            {
                "policy": policy_name,
                "AURS": area_under_reliability_surface(surface),
                "worst_case_value": worst_value,
                "worst_case_point": str(worst_point),
                "gated_success": gated_success_rate(policy_records),
                "corrupt_success": corrupt_success_rate(policy_records),
                "DCS": deferral_calibration_score(policy_records),
                "DCS_defer_precision": deferral_precision(policy_records),
                "DCS_defer_recall": deferral_recall(policy_records),
                "DCS_commit_precision": commit_precision(policy_records),
                "DCS_commit_recall": commit_recall(policy_records),
                "IER": irreversible_error_rate(policy_records),
                "EFV": evidence_freshness_violation_rate(policy_records),
                "over_deferral_rate": over_deferral_rate(policy_records),
                "premature_commit_rate": premature_commit_rate(policy_records),
                "turn_budget_exhaustion": turn_budget_exhaustion_rate(policy_records),
                "delayed_contradictions": delayed_contradiction_rate(policy_records),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "policy",
                "AURS",
                "worst_case_value",
                "worst_case_point",
                "gated_success",
                "corrupt_success",
                "DCS",
                "DCS_defer_precision",
                "DCS_defer_recall",
                "DCS_commit_precision",
                "DCS_commit_recall",
                "IER",
                "EFV",
                "over_deferral_rate",
                "premature_commit_rate",
                "turn_budget_exhaustion",
                "delayed_contradictions",
            ]
        )
    return pd.DataFrame(rows).sort_values("policy").reset_index(drop=True)

from pathlib import Path

import pandas as pd

from defer.core.interfaces import ReliabilityRecord
from defer.core.io import write_jsonl
from scripts.evaluate_metrics import run as evaluate_run


def _record(policy: str, seed: int) -> ReliabilityRecord:
    return ReliabilityRecord(
        episode_id=f"e_{policy}_{seed}",
        scenario_id="s1",
        domain="email",
        delay_mechanism="none",
        policy_name=policy,
        seed=seed,
        k=1,
        epsilon=0.1,
        lambda_fault=0.1,
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


def test_evaluate_metrics_surfaces_fallback_rate(tmp_path: Path) -> None:
    records_path = tmp_path / "reliability_records.jsonl"
    metrics_dir = tmp_path / "metrics"
    write_jsonl(
        records_path,
        [
            _record("defer_full", 42).model_dump(mode="json"),
            _record("runtime_verification_only", 42).model_dump(mode="json"),
        ],
    )
    (tmp_path / "fallback_metrics.csv").write_text(
        "\n".join(
            [
                "policy,total_decisions,parse_failures,fallback_calls,fallback_rate",
                "defer_full,100,4,4,0.04",
                "runtime_verification_only,100,0,0,0.0",
            ]
        ),
        encoding="utf-8",
    )

    evaluate_run(
        records_path=records_path,
        output_dir=metrics_dir,
        bootstrap_resamples=200,
        seed=42,
        min_episodes_per_cell=1,
        strict_coverage=False,
    )

    summary = pd.read_csv(metrics_dir / "summary_metrics.csv")
    assert "fallback_rate" in summary.columns
    defer_row = summary[summary["policy"] == "defer_full"].iloc[0]
    assert abs(float(defer_row["fallback_rate"]) - 0.04) < 1e-9
    assert int(defer_row["parse_failures"]) == 4

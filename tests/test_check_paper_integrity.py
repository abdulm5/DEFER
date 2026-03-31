from __future__ import annotations

import json
from pathlib import Path

from scripts.check_paper_integrity import run


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_check_paper_integrity_passes_for_complete_minimal_tree(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    protocol_path = tmp_path / "protocol.yaml"

    # Core files
    _write_json(run_dir / "checkpoint_eval/main/run_meta.json", {"ok": True})
    _write_json(run_dir / "checkpoint_eval/adversarial/run_meta.json", {"ok": True})
    _write_json(run_dir / "checkpoint_eval/theory/theory_comparison.json", {"ok": True})
    _write_json(run_dir / "checkpoint_eval/main_metrics/claim_gates.json", {"gate_1": True, "gate_2": True})
    (run_dir / "checkpoint_eval/main/reliability_records.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint_eval/main/reliability_records.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "human_eval/annotation_tasks.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "human_eval/annotation_tasks.jsonl").write_text("{}\n", encoding="utf-8")

    _write_csv(
        run_dir / "checkpoint_eval/main/fallback_metrics.csv",
        "policy,total_decisions,parse_failures,fallback_calls,fallback_rate",
        ["defer_full,10,0,0,0.0"],
    )
    _write_csv(
        run_dir / "checkpoint_eval/adversarial/fallback_metrics.csv",
        "policy,total_decisions,parse_failures,fallback_calls,fallback_rate",
        ["defer_full,10,0,0,0.0"],
    )
    _write_csv(
        run_dir / "checkpoint_eval/main_metrics/summary_metrics.csv",
        "policy,AURS,DCS",
        ["defer_full,0.5,0.5"],
    )
    _write_csv(
        run_dir / "checkpoint_eval/adversarial_metrics/summary_metrics.csv",
        "policy,AURS,DCS",
        ["defer_full,0.5,0.5"],
    )
    _write_csv(
        run_dir / "checkpoint_eval/main_metrics/cell_coverage.csv",
        "policy,domain,epsilon,lambda_fault,episode_count,meets_minimum",
        ["defer_full,calendar,0.0,0.0,1,True"],
    )
    _write_csv(
        run_dir / "checkpoint_eval/main_metrics/delay_mechanism_coverage.csv",
        "policy,delay_mechanism,episode_count,meets_minimum",
        ["defer_full,none,1,True"],
    )
    _write_csv(
        run_dir / "checkpoint_eval/main_metrics/pairwise_tests.csv",
        "metric,policy_a,policy_b,diff_point,ci_low,ci_high,p_value_two_sided,matched_clusters,adjusted_p_value",
        [
            "AURS,defer_full,runtime_verification_only,0.1,0.0,0.2,0.05,1,0.05",
            "DCS,defer_full,runtime_verification_only,0.1,0.0,0.2,0.05,1,0.05",
        ],
    )

    # Holdout files
    _write_json(run_dir / "checkpoint_eval_holdouts/delay/run_meta.json", {"ok": True})
    _write_csv(
        run_dir / "checkpoint_eval_holdouts/delay/fallback_metrics.csv",
        "policy,total_decisions,parse_failures,fallback_calls,fallback_rate",
        ["defer_full,10,0,0,0.0"],
    )
    _write_csv(
        run_dir / "checkpoint_eval_holdouts/delay_metrics/summary_metrics.csv",
        "policy,AURS,DCS",
        ["defer_full,0.5,0.5"],
    )
    for domain in [
        "calendar",
        "email",
        "rest",
        "sql",
        "webhook",
        "file_storage",
        "access_control",
        "notification",
    ]:
        (run_dir / f"checkpoint_eval_holdouts/domain_{domain}").mkdir(parents=True, exist_ok=True)
        (run_dir / f"checkpoint_eval_holdouts/domain_{domain}_metrics").mkdir(
            parents=True,
            exist_ok=True,
        )
        _write_json(
            run_dir / f"checkpoint_eval_holdouts/domain_{domain}/run_meta.json",
            {"ok": True},
        )
        _write_csv(
            run_dir / f"checkpoint_eval_holdouts/domain_{domain}/fallback_metrics.csv",
            "policy,total_decisions,parse_failures,fallback_calls,fallback_rate",
            ["defer_full,10,0,0,0.0"],
        )
        _write_csv(
            run_dir / f"checkpoint_eval_holdouts/domain_{domain}_metrics/summary_metrics.csv",
            "policy,AURS,DCS",
            ["defer_full,0.5,0.5"],
        )

    protocol_path.write_text(
        "\n".join(
            [
                "version: 1",
                "comparisons:",
                "  - [defer_full, runtime_verification_only]",
            ]
        ),
        encoding="utf-8",
    )

    run(
        run_dir=run_dir,
        protocol_path=protocol_path,
        max_fallback_rate=0.10,
        require_seed_sweep=False,
    )

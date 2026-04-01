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


def _create_minimal_primary_tree(run_dir: Path, protocol_path: Path) -> None:
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
        (run_dir / f"checkpoint_eval_holdouts/domain_{domain}_metrics").mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / f"checkpoint_eval_holdouts/domain_{domain}/run_meta.json", {"ok": True})
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
        "\n".join(["version: 1", "comparisons:", "  - [defer_full, runtime_verification_only]"]),
        encoding="utf-8",
    )


def _create_api_sota_tree(api_dir: Path, *, sampling_seed: int = 42, failed_job: bool = False) -> None:
    model_dir = api_dir / "gpt4o"
    out_dir = model_dir / "zero_shot"
    metrics_dir = model_dir / "zero_shot_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        out_dir / "run_meta.json",
        {"sampling_seed": sampling_seed, "scenarios": 100},
    )
    (out_dir / "reliability_records.jsonl").write_text("{}\n", encoding="utf-8")
    _write_csv(
        out_dir / "fallback_metrics.csv",
        "policy,total_decisions,parse_failures,fallback_calls,fallback_rate",
        ["frontier,10,0,0,0.0"],
    )
    _write_csv(
        metrics_dir / "pairwise_tests.csv",
        "metric,policy_a,policy_b,diff_point,ci_low,ci_high,p_value_two_sided,matched_clusters,adjusted_p_value",
        ["AURS,a,b,0,0,0,1,1,1"],
    )

    jobs = [
        {
            "model_id": "gpt4o",
            "model": "gpt-4o",
            "status": "failed" if failed_job else "success",
            "variants": [
                {
                    "variant": "zero_shot",
                    "output_dir": str(out_dir),
                    "run_meta": str(out_dir / "run_meta.json"),
                    "fallback_metrics": str(out_dir / "fallback_metrics.csv"),
                    "pairwise_tests": str(metrics_dir / "pairwise_tests.csv"),
                }
            ],
        }
    ]
    _write_json(api_dir / "matrix_manifest.json", {"jobs": jobs})
    _write_csv(
        api_dir / "matrix_summary.csv",
        "model_id,model,variant,status,scenario_count,sampling_seed,max_fallback_rate_observed,run_meta,fallback_metrics,pairwise_tests,error",
        [
            "gpt4o,gpt-4o,zero_shot,success,100,42,0.0,"
            f"{out_dir / 'run_meta.json'},{out_dir / 'fallback_metrics.csv'},{metrics_dir / 'pairwise_tests.csv'},"
        ],
    )


def test_check_paper_integrity_with_api_sota_dir_passes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    protocol_path = tmp_path / "protocol.yaml"
    api_dir = tmp_path / "api_sota"
    _create_minimal_primary_tree(run_dir, protocol_path)
    _create_api_sota_tree(api_dir, sampling_seed=42, failed_job=False)

    run(
        run_dir=run_dir,
        protocol_path=protocol_path,
        max_fallback_rate=0.10,
        require_seed_sweep=False,
        api_sota_dir=api_dir,
    )


def test_check_paper_integrity_with_api_sota_sampling_seed_mismatch_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    protocol_path = tmp_path / "protocol.yaml"
    api_dir = tmp_path / "api_sota"
    _create_minimal_primary_tree(run_dir, protocol_path)
    _create_api_sota_tree(api_dir, sampling_seed=42, failed_job=False)
    # Add second model with different sampling_seed.
    second = api_dir / "model2" / "zero_shot"
    second_metrics = api_dir / "model2" / "zero_shot_metrics"
    second.mkdir(parents=True, exist_ok=True)
    second_metrics.mkdir(parents=True, exist_ok=True)
    _write_json(second / "run_meta.json", {"sampling_seed": 99, "scenarios": 100})
    (second / "reliability_records.jsonl").write_text("{}\n", encoding="utf-8")
    _write_csv(
        second / "fallback_metrics.csv",
        "policy,total_decisions,parse_failures,fallback_calls,fallback_rate",
        ["frontier,10,0,0,0.0"],
    )
    _write_csv(
        second_metrics / "pairwise_tests.csv",
        "metric,policy_a,policy_b,diff_point,ci_low,ci_high,p_value_two_sided,matched_clusters,adjusted_p_value",
        ["AURS,a,b,0,0,0,1,1,1"],
    )
    manifest = json.loads((api_dir / "matrix_manifest.json").read_text(encoding="utf-8"))
    manifest["jobs"].append(
        {
            "model_id": "model2",
            "model": "x",
            "status": "success",
            "variants": [
                {
                    "variant": "zero_shot",
                    "output_dir": str(second),
                    "run_meta": str(second / "run_meta.json"),
                    "fallback_metrics": str(second / "fallback_metrics.csv"),
                    "pairwise_tests": str(second_metrics / "pairwise_tests.csv"),
                }
            ],
        }
    )
    (api_dir / "matrix_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    try:
        run(
            run_dir=run_dir,
            protocol_path=protocol_path,
            max_fallback_rate=0.10,
            require_seed_sweep=False,
            api_sota_dir=api_dir,
        )
        assert False, "Expected SystemExit for inconsistent sampling seeds"
    except SystemExit:
        pass


def test_check_paper_integrity_with_failed_api_job_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    protocol_path = tmp_path / "protocol.yaml"
    api_dir = tmp_path / "api_sota"
    _create_minimal_primary_tree(run_dir, protocol_path)
    _create_api_sota_tree(api_dir, sampling_seed=42, failed_job=True)

    try:
        run(
            run_dir=run_dir,
            protocol_path=protocol_path,
            max_fallback_rate=0.10,
            require_seed_sweep=False,
            api_sota_dir=api_dir,
        )
        assert False, "Expected SystemExit for failed API matrix jobs"
    except SystemExit:
        pass

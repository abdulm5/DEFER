from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.run_api_sota_matrix import run


def _write_matrix_config(path: Path, models: list[dict]) -> None:
    payload = {
        "defaults": {
            "variants": "artifacts/test_run_qwen25/data/variant_tasks.jsonl",
            "split": "test",
            "repeats": 1,
            "seed": 42,
            "sampling_seed": 42,
            "max_scenarios": 3,
            "include_baselines": ["runtime_verification_only"],
            "fallback_policy": "runtime_verification_only",
            "auth_mode": "api_key",
            "api_key_header": "api-key",
            "query_params": {"api-version": "2024-10-21"},
            "bootstrap_resamples": 10,
            "min_episodes_per_cell": 1,
            "strict_coverage": False,
        },
        "models": models,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _fake_run_api_eval(**kwargs) -> None:
    output_dir = Path(kwargs["output_dir"])
    if "bad_model" in str(output_dir):
        raise RuntimeError("simulated api failure")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reliability_records.jsonl").write_text("{}\n", encoding="utf-8")
    (output_dir / "episode_traces.jsonl").write_text("{}\n", encoding="utf-8")
    (output_dir / "fallback_metrics.csv").write_text(
        "policy,total_decisions,parse_failures,fallback_calls,fallback_rate\n"
        "test,10,0,0,0.0\n",
        encoding="utf-8",
    )
    (output_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "scenarios": kwargs["max_scenarios"],
                "sampling_seed": kwargs["sampling_seed"],
            }
        ),
        encoding="utf-8",
    )


def _fake_evaluate_metrics(**kwargs) -> None:
    output_dir = Path(kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pairwise_tests.csv").write_text(
        "metric,policy_a,policy_b,diff_point,ci_low,ci_high,p_value_two_sided,matched_clusters,adjusted_p_value\n"
        "AURS,a,b,0,0,0,1,1,1\n",
        encoding="utf-8",
    )
    (output_dir / "summary_metrics.csv").write_text(
        "policy,AURS,DCS\nx,0.0,0.0\n",
        encoding="utf-8",
    )


def test_matrix_runner_writes_manifest_and_summary(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "matrix.yaml"
    output_root = tmp_path / "api_sota"
    _write_matrix_config(
        config,
        models=[
            {
                "id": "model_a",
                "model": "gpt-4o",
                "base_url": "https://example.test/openai/v1/chat/completions",
                "api_key_env": "AZURE_OPENAI_API_KEY",
            },
            {
                "id": "model_b",
                "model": "sonnet",
                "base_url": "https://example2.test/openai/v1/chat/completions",
                "api_key_env": "AZURE_OPENAI_API_KEY",
            },
        ],
    )
    monkeypatch.setattr("scripts.run_api_sota_matrix.run_api_eval_run", _fake_run_api_eval)
    monkeypatch.setattr("scripts.run_api_sota_matrix.evaluate_metrics_run", _fake_evaluate_metrics)

    run(
        config=config,
        output_root=output_root,
        variants=None,
        workers=2,
        allow_partial=False,
    )

    manifest = json.loads((output_root / "matrix_manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(output_root / "matrix_summary.csv")
    assert manifest["failed_jobs"] == 0
    assert len(summary) == 4
    assert set(summary["status"]) == {"success"}


def test_matrix_runner_fails_without_allow_partial(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "matrix.yaml"
    output_root = tmp_path / "api_sota"
    _write_matrix_config(
        config,
        models=[
            {
                "id": "good_model",
                "model": "gpt-4o",
                "base_url": "https://example.test/openai/v1/chat/completions",
                "api_key_env": "AZURE_OPENAI_API_KEY",
            },
            {
                "id": "bad_model",
                "model": "gpt-5",
                "base_url": "https://example.test/openai/v1/chat/completions",
                "api_key_env": "AZURE_OPENAI_API_KEY",
            },
        ],
    )
    monkeypatch.setattr("scripts.run_api_sota_matrix.run_api_eval_run", _fake_run_api_eval)
    monkeypatch.setattr("scripts.run_api_sota_matrix.evaluate_metrics_run", _fake_evaluate_metrics)

    try:
        run(
            config=config,
            output_root=output_root,
            variants=None,
            workers=2,
            allow_partial=False,
        )
        assert False, "Expected SystemExit when allow_partial=False and a model fails"
    except SystemExit:
        pass

    manifest = json.loads((output_root / "matrix_manifest.json").read_text(encoding="utf-8"))
    assert manifest["failed_jobs"] == 1


def test_matrix_runner_allows_partial(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "matrix.yaml"
    output_root = tmp_path / "api_sota"
    _write_matrix_config(
        config,
        models=[
            {
                "id": "bad_model",
                "model": "gpt-5",
                "base_url": "https://example.test/openai/v1/chat/completions",
                "api_key_env": "AZURE_OPENAI_API_KEY",
            },
        ],
    )
    monkeypatch.setattr("scripts.run_api_sota_matrix.run_api_eval_run", _fake_run_api_eval)
    monkeypatch.setattr("scripts.run_api_sota_matrix.evaluate_metrics_run", _fake_evaluate_metrics)

    run(
        config=config,
        output_root=output_root,
        variants=None,
        workers=2,
        allow_partial=True,
    )

    summary = pd.read_csv(output_root / "matrix_summary.csv")
    assert set(summary["status"]) == {"failed"}

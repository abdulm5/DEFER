from __future__ import annotations

import argparse
import csv
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from defer.core.io import write_json
from scripts.evaluate_metrics import run as evaluate_metrics_run
from scripts.run_api_eval import run as run_api_eval_run


def _as_str_dict(payload: dict[str, Any] | None) -> dict[str, str]:
    if not payload:
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def _as_object_dict(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping for extra_body, got: {type(payload).__name__}")
    return {str(key): value for key, value in payload.items()}


def _as_string_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return default


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing matrix config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid matrix config at {path}; expected mapping root.")
    return payload


@dataclass
class MatrixRuntimeConfig:
    variants: Path
    output_root: Path
    split: str
    repeats: int
    seed: int
    sampling_seed: int
    max_scenarios: int
    include_baselines: list[str]
    fallback_policy: str
    max_new_tokens: int
    temperature: float
    top_p: float
    timeout_seconds: int
    auth_mode: str
    api_key_header: str
    extra_headers: dict[str, str]
    query_params: dict[str, str]
    extra_body: dict[str, Any]
    max_retries: int
    retry_backoff_seconds: float
    retry_max_backoff_seconds: float
    bootstrap_resamples: int
    min_episodes_per_cell: int
    strict_coverage: bool


def _load_runtime_config(
    config_payload: dict[str, Any],
    variants_override: Path | None,
    output_root: Path,
) -> MatrixRuntimeConfig:
    defaults = config_payload.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("Config key 'defaults' must be a mapping.")

    variants_raw = variants_override or Path(defaults.get("variants", "artifacts/final_run_v1/data/variant_tasks.jsonl"))
    sampling_seed_raw = defaults.get("sampling_seed")
    if sampling_seed_raw is None:
        raise ValueError(
            "Matrix config must set defaults.sampling_seed (fairness anchor across models)."
        )
    include_baselines = _as_string_list(defaults.get("include_baselines"), ["runtime_verification_only"])
    return MatrixRuntimeConfig(
        variants=Path(variants_raw),
        output_root=output_root,
        split=str(defaults.get("split", "test")),
        repeats=int(defaults.get("repeats", 3)),
        seed=int(defaults.get("seed", 42)),
        sampling_seed=int(sampling_seed_raw),
        max_scenarios=int(defaults.get("max_scenarios", 1000)),
        include_baselines=include_baselines,
        fallback_policy=str(defaults.get("fallback_policy", "runtime_verification_only")),
        max_new_tokens=int(defaults.get("max_new_tokens", 128)),
        temperature=float(defaults.get("temperature", 0.0)),
        top_p=float(defaults.get("top_p", 1.0)),
        timeout_seconds=int(defaults.get("timeout_seconds", 60)),
        auth_mode=str(defaults.get("auth_mode", "api_key")),
        api_key_header=str(defaults.get("api_key_header", "api-key")),
        extra_headers=_as_str_dict(defaults.get("extra_headers")),
        query_params=_as_str_dict(defaults.get("query_params")),
        extra_body=_as_object_dict(defaults.get("extra_body")),
        max_retries=int(defaults.get("max_retries", 3)),
        retry_backoff_seconds=float(defaults.get("retry_backoff_seconds", 1.0)),
        retry_max_backoff_seconds=float(defaults.get("retry_max_backoff_seconds", 8.0)),
        bootstrap_resamples=int(defaults.get("bootstrap_resamples", 5000)),
        min_episodes_per_cell=int(defaults.get("min_episodes_per_cell", 100)),
        strict_coverage=bool(defaults.get("strict_coverage", False)),
    )


def _fallback_max(path: Path) -> float:
    frame = pd.read_csv(path)
    if frame.empty:
        return 0.0
    return float(frame["fallback_rate"].max())


def _run_variant(
    *,
    runtime: MatrixRuntimeConfig,
    output_dir: Path,
    policy_name: str,
    model_name: str,
    base_url: str,
    api_key_env: str,
    auth_mode: str,
    api_key_header: str,
    extra_headers: dict[str, str],
    query_params: dict[str, str],
    extra_body: dict[str, Any],
    system_prompt_override: str | None,
    variant_label: str,
) -> dict[str, Any]:
    run_api_eval_run(
        variants=runtime.variants,
        output_dir=output_dir,
        split=runtime.split,
        repeats=runtime.repeats,
        seed=runtime.seed,
        sampling_seed=runtime.sampling_seed,
        max_scenarios=runtime.max_scenarios,
        api_policies=[f"{policy_name}={model_name}"],
        include_baselines=runtime.include_baselines,
        fallback_policy_name=runtime.fallback_policy,
        api_key_env=api_key_env,
        base_url=base_url,
        auth_mode=auth_mode,
        api_key_header=api_key_header,
        extra_headers=extra_headers,
        query_params=query_params,
        extra_body=extra_body,
        max_new_tokens=runtime.max_new_tokens,
        temperature=runtime.temperature,
        top_p=runtime.top_p,
        timeout_seconds=runtime.timeout_seconds,
        max_retries=runtime.max_retries,
        retry_backoff_seconds=runtime.retry_backoff_seconds,
        retry_max_backoff_seconds=runtime.retry_max_backoff_seconds,
        system_prompt_override=system_prompt_override,
    )
    metrics_dir = output_dir.parent / f"{output_dir.name}_metrics"
    evaluate_metrics_run(
        records_path=output_dir / "reliability_records.jsonl",
        output_dir=metrics_dir,
        bootstrap_resamples=runtime.bootstrap_resamples,
        seed=runtime.seed,
        min_episodes_per_cell=runtime.min_episodes_per_cell,
        strict_coverage=runtime.strict_coverage,
        fallback_metrics_path=output_dir / "fallback_metrics.csv",
    )
    run_meta = output_dir / "run_meta.json"
    run_meta_payload: dict[str, Any] = {}
    if run_meta.exists():
        run_meta_payload = json.loads(run_meta.read_text(encoding="utf-8"))
    return {
        "variant": variant_label,
        "output_dir": str(output_dir),
        "metrics_dir": str(metrics_dir),
        "run_meta": str(run_meta),
        "fallback_metrics": str(output_dir / "fallback_metrics.csv"),
        "pairwise_tests": str(metrics_dir / "pairwise_tests.csv"),
        "scenario_count": int(run_meta_payload.get("scenarios", 0)),
        "sampling_seed": runtime.sampling_seed,
        "max_fallback_rate_observed": _fallback_max(output_dir / "fallback_metrics.csv"),
    }


def _run_model_job(
    *,
    model_spec: dict[str, Any],
    runtime: MatrixRuntimeConfig,
) -> dict[str, Any]:
    if "model" not in model_spec:
        raise ValueError(f"Matrix model entry missing required 'model': {model_spec}")
    model_name = str(model_spec["model"])
    model_id = str(model_spec.get("id", model_name.replace("/", "_")))
    model_dir = runtime.output_root / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    base_url = str(model_spec.get("base_url"))
    api_key_env = str(model_spec.get("api_key_env"))
    if not base_url:
        raise ValueError(f"Model '{model_id}' missing required 'base_url'.")
    if not api_key_env:
        raise ValueError(f"Model '{model_id}' missing required 'api_key_env'.")

    auth_mode = str(model_spec.get("auth_mode", runtime.auth_mode))
    api_key_header = str(model_spec.get("api_key_header", runtime.api_key_header))
    extra_headers = dict(runtime.extra_headers)
    extra_headers.update(_as_str_dict(model_spec.get("extra_headers")))
    query_params = dict(runtime.query_params)
    query_params.update(_as_str_dict(model_spec.get("query_params")))
    extra_body = dict(runtime.extra_body)
    extra_body.update(_as_object_dict(model_spec.get("extra_body")))

    run_zero_shot = bool(model_spec.get("run_zero_shot", True))
    run_prompted = bool(model_spec.get("run_prompted_deferral", True))
    zero_shot_policy = str(model_spec.get("zero_shot_policy_name", f"{model_id}_zero_shot"))
    prompted_policy = str(model_spec.get("prompted_policy_name", "prompted_deferral"))

    variants: list[dict[str, Any]] = []
    if run_zero_shot:
        variants.append(
            _run_variant(
                runtime=runtime,
                output_dir=model_dir / "zero_shot",
                policy_name=zero_shot_policy,
                model_name=model_name,
                base_url=base_url,
                api_key_env=api_key_env,
                auth_mode=auth_mode,
                api_key_header=api_key_header,
                extra_headers=extra_headers,
                query_params=query_params,
                extra_body=extra_body,
                system_prompt_override=None,
                variant_label="zero_shot",
            )
        )
    if run_prompted:
        variants.append(
            _run_variant(
                runtime=runtime,
                output_dir=model_dir / "prompted_deferral",
                policy_name=prompted_policy,
                model_name=model_name,
                base_url=base_url,
                api_key_env=api_key_env,
                auth_mode=auth_mode,
                api_key_header=api_key_header,
                extra_headers=extra_headers,
                query_params=query_params,
                extra_body=extra_body,
                system_prompt_override="prompted_deferral",
                variant_label="prompted_deferral",
            )
        )

    return {
        "model_id": model_id,
        "model": model_name,
        "status": "success",
        "base_url": base_url,
        "api_key_env": api_key_env,
        "auth_mode": auth_mode,
        "api_key_header": api_key_header,
        "extra_headers": extra_headers,
        "query_params": query_params,
        "extra_body": extra_body,
        "variants": variants,
    }


def run(
    *,
    config: Path,
    output_root: Path,
    variants: Path | None,
    workers: int,
    allow_partial: bool,
) -> None:
    config_payload = _read_yaml(config)
    runtime = _load_runtime_config(
        config_payload=config_payload,
        variants_override=variants,
        output_root=output_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    model_specs = config_payload.get("models", [])
    if not isinstance(model_specs, list) or not model_specs:
        raise ValueError("Config must include a non-empty 'models' list.")

    jobs: list[dict[str, Any]] = []
    failed_jobs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_run_model_job, model_spec=spec, runtime=runtime): spec for spec in model_specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                jobs.append(future.result())
            except Exception as exc:  # noqa: BLE001
                model_id = str(spec.get("id", spec.get("model", "unknown")))
                failed_jobs.append(
                    {
                        "model_id": model_id,
                        "model": str(spec.get("model", "")),
                        "status": "failed",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

    jobs.extend(failed_jobs)
    jobs.sort(key=lambda row: str(row.get("model_id", "")))

    summary_rows: list[dict[str, Any]] = []
    for job in jobs:
        if job.get("status") != "success":
            summary_rows.append(
                {
                    "model_id": job.get("model_id", ""),
                    "model": job.get("model", ""),
                    "variant": "n/a",
                    "status": "failed",
                    "scenario_count": 0,
                    "sampling_seed": "",
                    "max_fallback_rate_observed": "",
                    "run_meta": "",
                    "fallback_metrics": "",
                    "pairwise_tests": "",
                    "error": job.get("error", ""),
                }
            )
            continue
        for variant in job.get("variants", []):
            summary_rows.append(
                {
                    "model_id": job.get("model_id", ""),
                    "model": job.get("model", ""),
                    "variant": variant.get("variant", ""),
                    "status": "success",
                    "scenario_count": variant.get("scenario_count", 0),
                    "sampling_seed": variant.get("sampling_seed", ""),
                    "max_fallback_rate_observed": variant.get("max_fallback_rate_observed", 0.0),
                    "run_meta": variant.get("run_meta", ""),
                    "fallback_metrics": variant.get("fallback_metrics", ""),
                    "pairwise_tests": variant.get("pairwise_tests", ""),
                    "error": "",
                }
            )

    summary_path = output_root / "matrix_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_id",
                "model",
                "variant",
                "status",
                "scenario_count",
                "sampling_seed",
                "max_fallback_rate_observed",
                "run_meta",
                "fallback_metrics",
                "pairwise_tests",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = {
        "config_path": str(config),
        "runtime_defaults": {
            "variants": str(runtime.variants),
            "split": runtime.split,
            "repeats": runtime.repeats,
            "seed": runtime.seed,
            "sampling_seed": runtime.sampling_seed,
            "max_scenarios": runtime.max_scenarios,
            "include_baselines": runtime.include_baselines,
            "fallback_policy": runtime.fallback_policy,
            "max_new_tokens": runtime.max_new_tokens,
            "temperature": runtime.temperature,
            "top_p": runtime.top_p,
            "timeout_seconds": runtime.timeout_seconds,
            "auth_mode": runtime.auth_mode,
            "api_key_header": runtime.api_key_header,
            "extra_headers": runtime.extra_headers,
            "query_params": runtime.query_params,
            "extra_body": runtime.extra_body,
            "max_retries": runtime.max_retries,
            "retry_backoff_seconds": runtime.retry_backoff_seconds,
            "retry_max_backoff_seconds": runtime.retry_max_backoff_seconds,
            "bootstrap_resamples": runtime.bootstrap_resamples,
            "min_episodes_per_cell": runtime.min_episodes_per_cell,
            "strict_coverage": runtime.strict_coverage,
            "workers": max(1, workers),
            "allow_partial": allow_partial,
        },
        "jobs": jobs,
        "summary_csv": str(summary_path),
        "failed_jobs": len([job for job in jobs if job.get("status") != "success"]),
    }
    write_json(output_root / "matrix_manifest.json", manifest)

    if failed_jobs and not allow_partial:
        raise SystemExit(f"{len(failed_jobs)} matrix jobs failed. See {output_root / 'matrix_manifest.json'}")
    print(f"Wrote matrix outputs under {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--variants",
        type=Path,
        default=None,
        help="Optional override for defaults.variants in config.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    run(
        config=args.config,
        output_root=args.output_root,
        variants=args.variants,
        workers=args.workers,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()

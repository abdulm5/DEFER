from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.evaluate_metrics import (
    _assert_protocol_pairwise_completeness,
    _load_protocol,
)

DOMAINS = [
    "calendar",
    "email",
    "rest",
    "sql",
    "webhook",
    "file_storage",
    "access_control",
    "notification",
]


def _require_file(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"Missing required file: {path}")


def _require_dir(path: Path, errors: list[str]) -> None:
    if not path.is_dir():
        errors.append(f"Missing required directory: {path}")


def _check_coverage_file(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"Missing coverage file: {path}")
        return
    frame = pd.read_csv(path)
    if "meets_minimum" not in frame.columns:
        errors.append(f"Coverage file missing 'meets_minimum' column: {path}")
        return
    failed = frame[~frame["meets_minimum"]]
    if not failed.empty:
        errors.append(f"Coverage failures in {path}: {len(failed)} rows below minimum.")


def _check_fallback_threshold(path: Path, threshold: float, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"Missing fallback metrics file: {path}")
        return
    frame = pd.read_csv(path)
    if frame.empty:
        errors.append(f"Fallback metrics file is empty: {path}")
        return
    if "fallback_rate" not in frame.columns:
        errors.append(f"Fallback metrics missing 'fallback_rate' column: {path}")
        return
    max_rate = float(frame["fallback_rate"].max())
    if max_rate > threshold:
        errors.append(
            f"Fallback threshold exceeded in {path}: max_fallback_rate={max_rate:.6f} > {threshold:.6f}"
        )


def _check_protocol_pairwise(
    pairwise_path: Path,
    protocol_path: Path,
    errors: list[str],
) -> None:
    if not pairwise_path.exists():
        errors.append(f"Missing pairwise tests file: {pairwise_path}")
        return
    pairwise_df = pd.read_csv(pairwise_path)
    protocol = _load_protocol(protocol_path)
    try:
        _assert_protocol_pairwise_completeness(
            pairwise_df=pairwise_df,
            protocol=protocol,
            required_metrics=("AURS", "DCS"),
        )
    except ValueError as exc:
        errors.append(str(exc))


def _check_api_sota_dir(
    api_sota_dir: Path,
    max_fallback_rate: float,
    errors: list[str],
) -> None:
    manifest_path = api_sota_dir / "matrix_manifest.json"
    summary_path = api_sota_dir / "matrix_summary.csv"
    _require_file(manifest_path, errors)
    _require_file(summary_path, errors)
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list) or not jobs:
        errors.append(f"API SOTA manifest has no jobs: {manifest_path}")
        return

    failed_jobs = [job for job in jobs if str(job.get("status")) != "success"]
    if failed_jobs:
        errors.append(
            f"API SOTA matrix contains failed jobs: {len(failed_jobs)} "
            f"(see {manifest_path})"
        )

    sampling_seeds: set[int] = set()
    scenario_counts: set[int] = set()
    for job in jobs:
        if str(job.get("status")) != "success":
            continue
        variants = job.get("variants", [])
        if not isinstance(variants, list) or not variants:
            errors.append(f"API SOTA job has no variants: {job.get('model_id')}")
            continue
        for variant in variants:
            run_meta = Path(str(variant.get("run_meta", "")))
            fallback_metrics = Path(str(variant.get("fallback_metrics", "")))
            pairwise_tests = Path(str(variant.get("pairwise_tests", "")))
            reliability_records = Path(str(variant.get("output_dir", ""))) / "reliability_records.jsonl"

            _require_file(run_meta, errors)
            _require_file(reliability_records, errors)
            _require_file(fallback_metrics, errors)
            _require_file(pairwise_tests, errors)

            if fallback_metrics.exists():
                _check_fallback_threshold(fallback_metrics, threshold=max_fallback_rate, errors=errors)

            if run_meta.exists():
                payload = json.loads(run_meta.read_text(encoding="utf-8"))
                if "sampling_seed" in payload:
                    try:
                        sampling_seeds.add(int(payload["sampling_seed"]))
                    except (TypeError, ValueError):
                        errors.append(f"Invalid sampling_seed in {run_meta}: {payload.get('sampling_seed')}")
                if "scenarios" in payload:
                    try:
                        scenario_counts.add(int(payload["scenarios"]))
                    except (TypeError, ValueError):
                        errors.append(f"Invalid scenarios count in {run_meta}: {payload.get('scenarios')}")

    if len(sampling_seeds) > 1:
        errors.append(
            f"Inconsistent API SOTA sampling_seed values: {sorted(sampling_seeds)}"
        )
    if len(scenario_counts) > 1:
        errors.append(
            f"Inconsistent API SOTA scenario counts across models: {sorted(scenario_counts)}"
        )


def run(
    run_dir: Path,
    protocol_path: Path,
    max_fallback_rate: float,
    require_seed_sweep: bool,
    api_sota_dir: Path | None = None,
) -> None:
    errors: list[str] = []

    # Core outputs
    _require_file(run_dir / "checkpoint_eval/main/reliability_records.jsonl", errors)
    _require_file(run_dir / "checkpoint_eval/main/fallback_metrics.csv", errors)
    _require_file(run_dir / "checkpoint_eval/main/run_meta.json", errors)
    _require_file(run_dir / "checkpoint_eval/main_metrics/summary_metrics.csv", errors)
    _require_file(run_dir / "checkpoint_eval/main_metrics/claim_gates.json", errors)
    _require_file(run_dir / "checkpoint_eval/main_metrics/pairwise_tests.csv", errors)
    _require_file(run_dir / "checkpoint_eval/adversarial/run_meta.json", errors)
    _require_file(run_dir / "checkpoint_eval/adversarial/fallback_metrics.csv", errors)
    _require_file(run_dir / "checkpoint_eval/adversarial_metrics/summary_metrics.csv", errors)
    _require_file(run_dir / "checkpoint_eval/theory/theory_comparison.json", errors)
    _require_file(run_dir / "human_eval/annotation_tasks.jsonl", errors)

    # Holdout outputs
    _require_file(run_dir / "checkpoint_eval_holdouts/delay/run_meta.json", errors)
    _require_file(run_dir / "checkpoint_eval_holdouts/delay/fallback_metrics.csv", errors)
    _require_file(run_dir / "checkpoint_eval_holdouts/delay_metrics/summary_metrics.csv", errors)
    for domain in DOMAINS:
        _require_dir(run_dir / f"checkpoint_eval_holdouts/domain_{domain}", errors)
        _require_dir(run_dir / f"checkpoint_eval_holdouts/domain_{domain}_metrics", errors)
        _require_file(run_dir / f"checkpoint_eval_holdouts/domain_{domain}/run_meta.json", errors)
        _require_file(
            run_dir / f"checkpoint_eval_holdouts/domain_{domain}/fallback_metrics.csv",
            errors,
        )
        _require_file(
            run_dir / f"checkpoint_eval_holdouts/domain_{domain}_metrics/summary_metrics.csv",
            errors,
        )

    # Coverage checks (main always required)
    _check_coverage_file(run_dir / "checkpoint_eval/main_metrics/cell_coverage.csv", errors)
    _check_coverage_file(run_dir / "checkpoint_eval/main_metrics/delay_mechanism_coverage.csv", errors)

    # Optional seed sweep checks
    seed_sweep_dir = run_dir / "checkpoint_eval/seed_sweep"
    seed_sweep_metrics_dir = run_dir / "checkpoint_eval/seed_sweep_metrics"
    if require_seed_sweep or seed_sweep_metrics_dir.exists():
        _require_file(seed_sweep_dir / "fallback_metrics.csv", errors)
        _require_file(seed_sweep_metrics_dir / "claim_gates.json", errors)
        _require_file(seed_sweep_metrics_dir / "summary_metrics.csv", errors)
        _check_coverage_file(seed_sweep_metrics_dir / "cell_coverage.csv", errors)
        _check_coverage_file(seed_sweep_metrics_dir / "delay_mechanism_coverage.csv", errors)

    # Protocol pairwise completeness
    _check_protocol_pairwise(
        pairwise_path=run_dir / "checkpoint_eval/main_metrics/pairwise_tests.csv",
        protocol_path=protocol_path,
        errors=errors,
    )

    # Fallback thresholds
    fallback_paths = sorted((run_dir / "checkpoint_eval").glob("**/fallback_metrics.csv"))
    fallback_paths.extend(
        sorted((run_dir / "checkpoint_eval_holdouts").glob("**/fallback_metrics.csv"))
    )
    for path in fallback_paths:
        _check_fallback_threshold(path, threshold=max_fallback_rate, errors=errors)

    if api_sota_dir is not None:
        _check_api_sota_dir(
            api_sota_dir=api_sota_dir,
            max_fallback_rate=max_fallback_rate,
            errors=errors,
        )

    if errors:
        print("Paper integrity check failed:")
        for err in errors:
            print(f" - {err}")
        raise SystemExit(1)
    print(f"Paper integrity check passed for {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-path",
        type=Path,
        default=Path("defer/configs/eval_protocol.yaml"),
    )
    parser.add_argument("--max-fallback-rate", type=float, default=0.10)
    parser.add_argument("--require-seed-sweep", action="store_true")
    parser.add_argument(
        "--api-sota-dir",
        type=Path,
        default=None,
        help="Optional supplementary API SOTA matrix directory to validate.",
    )
    args = parser.parse_args()
    run(
        run_dir=args.run_dir,
        protocol_path=args.protocol_path,
        max_fallback_rate=args.max_fallback_rate,
        require_seed_sweep=args.require_seed_sweep,
        api_sota_dir=args.api_sota_dir,
    )


if __name__ == "__main__":
    main()

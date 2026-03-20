from __future__ import annotations

import argparse
import hashlib
import statistics
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import yaml

from defer.analysis.tables import summary_table
from defer.core.interfaces import ReliabilityRecord
from defer.core.io import read_jsonl, write_json
from defer.metrics.deferral import (
    commit_precision,
    commit_recall,
    deferral_calibration_score,
    deferral_precision,
    deferral_recall,
)
from defer.metrics.reliability import area_under_reliability_surface, reliability_surface
from defer.metrics.stats import cluster_bootstrap_ci, paired_cluster_bootstrap_diff

def run(
    records_path: Path,
    output_dir: Path,
    bootstrap_resamples: int,
    seed: int,
    min_episodes_per_cell: int = 100,
    strict_coverage: bool = False,
    protocol_path: Path = Path("defer/configs/eval_protocol.yaml"),
    fallback_metrics_path: Path | None = None,
) -> None:
    rows = read_jsonl(records_path)
    records = [ReliabilityRecord(**row) for row in rows]
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol_payload = _load_protocol(protocol_path)
    write_json(output_dir / "evaluation_protocol_snapshot.json", protocol_payload)
    fallback_df = _load_fallback_metrics(
        records_path=records_path,
        fallback_metrics_path=fallback_metrics_path,
    )

    table = summary_table(records)
    table = _attach_fallback_metrics(table, fallback_df)
    table.to_csv(output_dir / "summary_metrics.csv", index=False)
    _write_breakdown(
        records=records,
        output_dir=output_dir,
        field_name="scenario_category",
        filename="category_summary.csv",
        fallback_df=fallback_df,
    )
    _write_breakdown(
        records=records,
        output_dir=output_dir,
        field_name="domain",
        filename="domain_summary.csv",
        fallback_df=fallback_df,
    )
    _write_breakdown(
        records=records,
        output_dir=output_dir,
        field_name="delay_mechanism",
        filename="delay_mechanism_summary.csv",
        fallback_df=fallback_df,
    )
    _write_seed_variance(records=records, output_dir=output_dir)
    _write_cell_coverage(
        records=records,
        output_dir=output_dir,
        min_episodes_per_cell=min_episodes_per_cell,
        strict_coverage=strict_coverage,
    )

    if not records:
        pd.DataFrame(
            columns=[
                "policy",
                "AURS",
                "AURS_ci_low",
                "AURS_ci_high",
                "DCS",
                "DCS_ci_low",
                "DCS_ci_high",
                "DCS_defer_precision",
                "DCS_defer_recall",
                "DCS_commit_precision",
                "DCS_commit_recall",
                "fallback_rate",
                "parse_failures",
                "fallback_calls",
                "total_decisions",
                "surface_points",
            ]
        ).to_csv(output_dir / "bootstrap_ci.csv", index=False)
        write_json(
            output_dir / "claim_gates.json",
            {"gate_1": False, "gate_2": False, "reason": "No records available."},
        )
        pd.DataFrame(
            columns=[
                "metric",
                "policy_a",
                "policy_b",
                "diff_point",
                "ci_low",
                "ci_high",
                "p_value_two_sided",
                "matched_clusters",
            ]
        ).to_csv(output_dir / "pairwise_tests.csv", index=False)
        print(f"Wrote metrics to {output_dir} (empty records)")
        return

    by_policy: dict[str, list[ReliabilityRecord]] = {}
    for record in records:
        by_policy.setdefault(record.policy_name, []).append(record)

    ci_rows: list[dict] = []
    for policy, policy_records in sorted(by_policy.items()):
        surface = reliability_surface(policy_records).get(policy, {})

        def aurs_fn(rs: list[ReliabilityRecord], policy_name: str = policy) -> float:
            policy_surface = reliability_surface(rs).get(policy_name, {})
            return area_under_reliability_surface(policy_surface)

        def cluster_key(record: ReliabilityRecord) -> tuple[int, str]:
            return (record.seed, record.scenario_id)

        dcs_fn = deferral_calibration_score
        dcs_def_precision_fn = deferral_precision
        dcs_def_recall_fn = deferral_recall
        dcs_commit_precision_fn = commit_precision
        dcs_commit_recall_fn = commit_recall
        aurs_seed_values = _metric_by_seed(policy_records, metric_fn=aurs_fn)
        dcs_seed_values = _metric_by_seed(policy_records, metric_fn=dcs_fn)
        aurs_point, aurs_lo, aurs_hi = cluster_bootstrap_ci(
            policy_records,
            metric_fn=aurs_fn,
            cluster_key_fn=cluster_key,
            n_resamples=bootstrap_resamples,
            seed=seed,
        )
        dcs_point, dcs_lo, dcs_hi = cluster_bootstrap_ci(
            policy_records,
            metric_fn=dcs_fn,
            cluster_key_fn=cluster_key,
            n_resamples=bootstrap_resamples,
            seed=seed + 1,
        )
        dcs_def_precision, _, _ = cluster_bootstrap_ci(
            policy_records,
            metric_fn=dcs_def_precision_fn,
            cluster_key_fn=cluster_key,
            n_resamples=max(250, bootstrap_resamples // 5),
            seed=seed + 2,
        )
        dcs_def_recall, _, _ = cluster_bootstrap_ci(
            policy_records,
            metric_fn=dcs_def_recall_fn,
            cluster_key_fn=cluster_key,
            n_resamples=max(250, bootstrap_resamples // 5),
            seed=seed + 3,
        )
        dcs_commit_precision_value, _, _ = cluster_bootstrap_ci(
            policy_records,
            metric_fn=dcs_commit_precision_fn,
            cluster_key_fn=cluster_key,
            n_resamples=max(250, bootstrap_resamples // 5),
            seed=seed + 4,
        )
        dcs_commit_recall_value, _, _ = cluster_bootstrap_ci(
            policy_records,
            metric_fn=dcs_commit_recall_fn,
            cluster_key_fn=cluster_key,
            n_resamples=max(250, bootstrap_resamples // 5),
            seed=seed + 5,
        )
        ci_rows.append(
            {
                "policy": policy,
                "AURS": aurs_point,
                "AURS_ci_low": aurs_lo,
                "AURS_ci_high": aurs_hi,
                "AURS_std": _std(aurs_seed_values),
                "DCS": dcs_point,
                "DCS_ci_low": dcs_lo,
                "DCS_ci_high": dcs_hi,
                "DCS_std": _std(dcs_seed_values),
                "DCS_defer_precision": dcs_def_precision,
                "DCS_defer_recall": dcs_def_recall,
                "DCS_commit_precision": dcs_commit_precision_value,
                "DCS_commit_recall": dcs_commit_recall_value,
                "surface_points": len(surface),
            }
        )

    ci_df = pd.DataFrame(ci_rows).sort_values("policy")
    ci_df = _attach_fallback_metrics(ci_df, fallback_df)
    ci_df.to_csv(output_dir / "bootstrap_ci.csv", index=False)

    claim_gate = _compute_claim_gates(ci_df)
    write_json(output_dir / "claim_gates.json", claim_gate)
    pairwise_df = _pairwise_significance(records=records, bootstrap_resamples=bootstrap_resamples, seed=seed)
    pairwise_df.to_csv(output_dir / "pairwise_tests.csv", index=False)
    print(f"Wrote metrics to {output_dir}")


def _compute_claim_gates(ci_df: pd.DataFrame) -> dict:
    row_defer = ci_df[ci_df["policy"] == "defer_full"]
    row_runtime = ci_df[ci_df["policy"] == "runtime_verification_only"]
    row_perfect = ci_df[ci_df["policy"] == "perfect_verifier_posttrain"]
    if row_defer.empty or row_runtime.empty or row_perfect.empty:
        return {"gate_1": False, "gate_2": False, "reason": "Missing required baselines."}

    defer = row_defer.iloc[0]
    runtime = row_runtime.iloc[0]
    perfect = row_perfect.iloc[0]

    gate_1 = (defer["AURS_ci_low"] > runtime["AURS_ci_high"]) and (
        defer["DCS_ci_low"] > runtime["DCS_ci_high"]
    )
    gate_2 = defer["DCS_ci_low"] > perfect["DCS_ci_high"]
    return {
        "gate_1": bool(gate_1),
        "gate_2": bool(gate_2),
        "gate_criteria": {
            "gate_1": "defer(AURS_ci_low) > runtime(AURS_ci_high) and defer(DCS_ci_low) > runtime(DCS_ci_high)",
            "gate_2": "defer(DCS_ci_low) > perfect_verifier(DCS_ci_high)",
        },
        "defer": defer.to_dict(),
        "runtime": runtime.to_dict(),
        "perfect_verifier": perfect.to_dict(),
    }


def _write_breakdown(
    records: list[ReliabilityRecord],
    output_dir: Path,
    field_name: str,
    filename: str,
    fallback_df: pd.DataFrame | None = None,
) -> None:
    if not records:
        pd.DataFrame(
            columns=[
                field_name,
                "policy",
                "AURS",
                "DCS",
                "fallback_rate",
                "parse_failures",
                "fallback_calls",
                "total_decisions",
            ]
        ).to_csv(output_dir / filename, index=False)
        return
    rows: list[pd.DataFrame] = []
    values = sorted({getattr(record, field_name, "unknown") for record in records})
    for value in values:
        subset = [record for record in records if getattr(record, field_name, "unknown") == value]
        frame = summary_table(subset)
        frame = _attach_fallback_metrics(frame, fallback_df)
        frame.insert(0, field_name, value)
        rows.append(frame)
    pd.concat(rows, ignore_index=True).to_csv(output_dir / filename, index=False)


def _metric_by_seed(
    records: list[ReliabilityRecord],
    metric_fn: Callable[[list[ReliabilityRecord]], float],
) -> list[float]:
    by_seed: dict[int, list[ReliabilityRecord]] = {}
    for record in records:
        by_seed.setdefault(record.seed, []).append(record)
    return [metric_fn(seed_records) for _, seed_records in sorted(by_seed.items())]


def _std(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) <= 1:
        return 0.0
    return float(statistics.stdev(vals))


def _write_seed_variance(records: list[ReliabilityRecord], output_dir: Path) -> None:
    if not records:
        pd.DataFrame(columns=["policy", "seed", "AURS", "DCS"]).to_csv(
            output_dir / "seed_metrics.csv", index=False
        )
        return
    by_policy: dict[str, list[ReliabilityRecord]] = {}
    for record in records:
        by_policy.setdefault(record.policy_name, []).append(record)
    rows: list[dict] = []
    for policy, policy_records in sorted(by_policy.items()):
        by_seed: dict[int, list[ReliabilityRecord]] = {}
        for record in policy_records:
            by_seed.setdefault(record.seed, []).append(record)
        for policy_seed, seed_records in sorted(by_seed.items()):
            seed_surface = reliability_surface(seed_records).get(policy, {})
            rows.append(
                {
                    "policy": policy,
                    "seed": policy_seed,
                    "AURS": area_under_reliability_surface(seed_surface),
                    "DCS": deferral_calibration_score(seed_records),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "seed_metrics.csv", index=False)


def _write_cell_coverage(
    records: list[ReliabilityRecord],
    output_dir: Path,
    min_episodes_per_cell: int,
    strict_coverage: bool,
) -> None:
    if not records:
        pd.DataFrame(
            columns=["policy", "domain", "epsilon", "lambda_fault", "episode_count", "meets_minimum"]
        ).to_csv(output_dir / "cell_coverage.csv", index=False)
        return
    df = pd.DataFrame([record.model_dump(mode="json") for record in records])
    coverage = (
        df.groupby(["policy_name", "domain", "epsilon", "lambda_fault"], as_index=False)
        .size()
        .rename(columns={"policy_name": "policy", "size": "episode_count"})
    )
    coverage["meets_minimum"] = coverage["episode_count"] >= min_episodes_per_cell
    coverage.to_csv(output_dir / "cell_coverage.csv", index=False)
    if strict_coverage and not bool(coverage["meets_minimum"].all()):
        failed = coverage[~coverage["meets_minimum"]]
        raise ValueError(
            "Coverage check failed: "
            f"{len(failed)} cells below minimum {min_episodes_per_cell} episodes."
        )


def _pairwise_significance(
    records: list[ReliabilityRecord],
    bootstrap_resamples: int,
    seed: int,
) -> pd.DataFrame:
    by_policy: dict[str, list[ReliabilityRecord]] = {}
    for record in records:
        by_policy.setdefault(record.policy_name, []).append(record)

    comparisons = [
        ("defer_full", "clean_sft_only"),
        ("defer_full", "runtime_verification_only"),
        ("defer_full", "perfect_verifier_posttrain"),
        ("defer_full", "success_signal_posttrain"),
        ("defer_full", "prompted_deferral"),
        ("success_signal_posttrain", "clean_sft_only"),
    ]

    def cluster_key(record: ReliabilityRecord) -> tuple[int, str]:
        return (record.seed, record.scenario_id)

    def aurs_metric(rs: list[ReliabilityRecord]) -> float:
        surfaces = reliability_surface(rs)
        if not surfaces:
            return 0.0
        _, surface = next(iter(surfaces.items()))
        return area_under_reliability_surface(surface)

    metric_map: dict[str, Callable[[list[ReliabilityRecord]], float]] = {
        "AURS": aurs_metric,
        "DCS": deferral_calibration_score,
    }

    rows: list[dict] = []
    for policy_a, policy_b in comparisons:
        if policy_a not in by_policy or policy_b not in by_policy:
            continue
        records_a = by_policy[policy_a]
        records_b = by_policy[policy_b]
        for metric_name, metric_fn in metric_map.items():
            result = paired_cluster_bootstrap_diff(
                records_a=records_a,
                records_b=records_b,
                metric_fn=metric_fn,
                cluster_key_fn=cluster_key,
                n_resamples=bootstrap_resamples,
                seed=seed + (1 if metric_name == "AURS" else 2),
            )
            rows.append(
                {
                    "metric": metric_name,
                    "policy_a": policy_a,
                    "policy_b": policy_b,
                    **result,
                }
            )
    return pd.DataFrame(rows)


def _load_protocol(protocol_path: Path) -> dict:
    if not protocol_path.exists():
        return {
            "path": str(protocol_path),
            "exists": False,
            "sha1": None,
            "payload": {},
        }
    text = protocol_path.read_text(encoding="utf-8")
    sha = hashlib.sha1(text.encode("utf-8")).hexdigest()
    payload = yaml.safe_load(text) or {}
    return {
        "path": str(protocol_path),
        "exists": True,
        "sha1": sha,
        "payload": payload,
    }


def _load_fallback_metrics(
    records_path: Path,
    fallback_metrics_path: Path | None,
) -> pd.DataFrame:
    candidate = fallback_metrics_path
    if candidate is None:
        default_candidate = records_path.parent / "fallback_metrics.csv"
        if default_candidate.exists():
            candidate = default_candidate
    if candidate is None or not candidate.exists():
        return pd.DataFrame(
            columns=[
                "policy",
                "fallback_rate",
                "parse_failures",
                "fallback_calls",
                "total_decisions",
            ]
        )
    frame = pd.read_csv(candidate)
    rename_map = {}
    if "policy_name" in frame.columns and "policy" not in frame.columns:
        rename_map["policy_name"] = "policy"
    frame = frame.rename(columns=rename_map)
    required = {"policy", "fallback_rate"}
    if not required.issubset(set(frame.columns)):
        return pd.DataFrame(
            columns=[
                "policy",
                "fallback_rate",
                "parse_failures",
                "fallback_calls",
                "total_decisions",
            ]
        )
    for col in ["parse_failures", "fallback_calls", "total_decisions"]:
        if col not in frame.columns:
            frame[col] = 0
    return frame[["policy", "fallback_rate", "parse_failures", "fallback_calls", "total_decisions"]]


def _attach_fallback_metrics(
    frame: pd.DataFrame,
    fallback_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if frame.empty:
        out = frame.copy()
        for col in ["fallback_rate", "parse_failures", "fallback_calls", "total_decisions"]:
            if col not in out.columns:
                out[col] = 0.0 if col == "fallback_rate" else 0
        return out
    if fallback_df is None or fallback_df.empty:
        out = frame.copy()
        out["fallback_rate"] = 0.0
        out["parse_failures"] = 0
        out["fallback_calls"] = 0
        out["total_decisions"] = 0
        return out
    out = frame.merge(fallback_df, on="policy", how="left")
    out["fallback_rate"] = out["fallback_rate"].fillna(0.0)
    out["parse_failures"] = out["parse_failures"].fillna(0).astype(int)
    out["fallback_calls"] = out["fallback_calls"].fillna(0).astype(int)
    out["total_decisions"] = out["total_decisions"].fillna(0).astype(int)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-path", type=Path, default=Path("artifacts/runs/reliability_records.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/metrics"))
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-episodes-per-cell", type=int, default=100)
    parser.add_argument("--strict-coverage", action="store_true")
    parser.add_argument(
        "--protocol-path",
        type=Path,
        default=Path("defer/configs/eval_protocol.yaml"),
    )
    parser.add_argument(
        "--fallback-metrics-path",
        type=Path,
        default=None,
        help="Optional path to fallback_metrics.csv; defaults to sibling of records file when present.",
    )
    args = parser.parse_args()
    run(
        records_path=args.records_path,
        output_dir=args.output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        min_episodes_per_cell=args.min_episodes_per_cell,
        strict_coverage=args.strict_coverage,
        protocol_path=args.protocol_path,
        fallback_metrics_path=args.fallback_metrics_path,
    )

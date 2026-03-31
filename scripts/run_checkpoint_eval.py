from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path

from defer.baselines.model_policy import HFCheckpointPolicy, InferenceConfig
from defer.baselines.policies import policy_registry
from defer.baselines.runner import RunnerConfig, run_policies
from defer.core.io import read_jsonl, write_json, write_jsonl
from defer.data.schema import VariantTask
from defer.sim.scenario import Scenario
from defer.sim.sampling import deterministic_sample_scenarios


def _assert_no_contamination(training_traces_path: Path, eval_scenarios: list) -> None:
    """Raise if any eval scenario_id appears in training traces."""
    if not training_traces_path.exists():
        raise FileNotFoundError(
            f"Training traces path not found: {training_traces_path}. "
            "Cannot verify train/test separation."
        )
    train_ids: set[str] = set()
    for row in read_jsonl(training_traces_path):
        train_ids.add(row["scenario_id"])
    eval_ids = {s.scenario_id for s in eval_scenarios}
    overlap = train_ids & eval_ids
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(
            f"Train/test contamination detected: {len(overlap)} scenario_ids overlap. "
            f"Examples: {examples}"
        )


def _validate_fallback_rates(fallback_rows: list[dict], max_rate: float) -> None:
    """Raise if any policy exceeds the maximum allowed fallback rate."""
    for row in fallback_rows:
        rate = float(row.get("fallback_rate", 0.0))
        if math.isnan(rate) or rate > max_rate:
            raise ValueError(
                f"Policy '{row['policy']}' fallback rate {rate} exceeds threshold {max_rate}"
            )


def _rows_to_scenarios(
    rows: list[dict],
    split: str | None,
    domains: set[str] | None = None,
    include_delay_mechanisms: set[str] | None = None,
    exclude_delay_mechanisms: set[str] | None = None,
) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for row in rows:
        task = VariantTask(**row)
        if split is not None and task.split != split:
            continue
        if domains is not None and task.domain not in domains:
            continue
        if include_delay_mechanisms is not None and task.delay_mechanism not in include_delay_mechanisms:
            continue
        if exclude_delay_mechanisms is not None and task.delay_mechanism in exclude_delay_mechanisms:
            continue
        scenarios.append(
            Scenario(
                scenario_id=task.variant_id,
                domain=task.domain,
                prompt=task.prompt,
                required_tool=task.required_tool,
                tool_args=task.tool_args,
                expects_irreversible=task.expects_irreversible,
                requires_refresh=task.requires_refresh,
                has_delayed_truth=task.delay_setting == "delayed",
                delayed_truth_category=task.delayed_truth_category,
                delay_mechanism=task.delay_mechanism,
                metadata={
                    "epsilon": task.epsilon,
                    "lambda_fault": task.lambda_fault,
                    "delayed_truth_category": task.delayed_truth_category,
                    "delay_mechanism": task.delay_mechanism,
                },
            )
        )
    return scenarios


def _parse_model_specs(items: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --model-policy '{item}'. Expected format policy_name=/path/to/checkpoint"
            )
        name, path = item.split("=", 1)
        name = name.strip()
        checkpoint_path = path.strip()
        if not name:
            raise ValueError(f"Invalid --model-policy '{item}': missing policy name.")
        specs.append((name, checkpoint_path))
    return specs


def _release_model_policy(policy: HFCheckpointPolicy) -> None:
    for attr in ("model", "tokenizer"):
        if hasattr(policy, attr):
            delattr(policy, attr)
    gc.collect()
    try:
        import torch
    except Exception:  # pragma: no cover - torch may not be installed for non-checkpoint paths
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(
    variants: Path,
    output_dir: Path,
    split: str,
    repeats: int,
    seed: int,
    max_scenarios: int,
    model_policies: list[str],
    include_baselines: list[str] | None = None,
    fallback_policy_name: str = "runtime_verification_only",
    domains: set[str] | None = None,
    include_delay_mechanisms: set[str] | None = None,
    exclude_delay_mechanisms: set[str] | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    training_traces: Path | None = None,
    max_fallback_rate: float | None = None,
    sampling_seed: int | None = None,
) -> None:
    if not model_policies:
        raise ValueError("At least one --model-policy name=checkpoint_path is required.")

    rows = read_jsonl(variants)
    scenarios = _rows_to_scenarios(
        rows,
        split=split,
        domains=domains,
        include_delay_mechanisms=include_delay_mechanisms,
        exclude_delay_mechanisms=exclude_delay_mechanisms,
    )
    if not scenarios:
        raise ValueError(
            "No scenarios found for requested split/filter combination: "
            f"split={split}, domains={sorted(domains) if domains else 'all'}, "
            "include_delay_mechanisms="
            f"{sorted(include_delay_mechanisms) if include_delay_mechanisms else 'all'}, "
            "exclude_delay_mechanisms="
            f"{sorted(exclude_delay_mechanisms) if exclude_delay_mechanisms else []}."
        )
    scenarios = deterministic_sample_scenarios(
        scenarios=scenarios,
        max_scenarios=max_scenarios,
        seed=sampling_seed if sampling_seed is not None else seed,
        salt=f"run_checkpoint_eval::{split}",
    )

    if training_traces is not None:
        _assert_no_contamination(training_traces, scenarios)

    registry = policy_registry()
    if fallback_policy_name not in registry:
        raise ValueError(f"Unknown fallback policy: {fallback_policy_name}")

    policy_names_for_fallback: list[str] = []
    baseline_names = include_baselines or []
    baseline_policies = []
    for baseline_name in baseline_names:
        if baseline_name not in registry:
            raise ValueError(f"Unknown baseline policy in --include-baselines: {baseline_name}")
        baseline_policies.append(registry[baseline_name])
        policy_names_for_fallback.append(baseline_name)

    traces = []
    records = []
    if baseline_policies:
        baseline_traces, baseline_records = run_policies(
            scenarios=scenarios,
            policies=baseline_policies,
            config=RunnerConfig(repeats=repeats, seed=seed),
        )
        traces.extend(baseline_traces)
        records.extend(baseline_records)

    model_specs = _parse_model_specs(model_policies)
    model_stats: dict[str, dict] = {}
    for name, checkpoint_path in model_specs:
        fallback = registry[fallback_policy_name]
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint path for policy '{name}' does not exist: {checkpoint_path}"
            )
        policy_names_for_fallback.append(name)
        model_stats[name] = {
            "checkpoint_path": str(checkpoint_path),
            "fallback_policy": fallback.name,
        }
        policy = HFCheckpointPolicy(
            name=name,
            checkpoint_path=checkpoint_path,
            fallback_policy=fallback,
            inference=InferenceConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
        )
        try:
            model_traces, model_records = run_policies(
                scenarios=scenarios,
                policies=[policy],
                config=RunnerConfig(repeats=repeats, seed=seed),
            )
            traces.extend(model_traces)
            records.extend(model_records)
            model_stats[name]["inference_stats"] = policy.stats()
        finally:
            _release_model_policy(policy)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "episode_traces.jsonl", [trace.model_dump(mode="json") for trace in traces])
    write_jsonl(
        output_dir / "reliability_records.jsonl",
        [record.model_dump(mode="json") for record in records],
    )

    decision_counts: dict[str, int] = {}
    for trace in traces:
        decision_counts[trace.policy_name] = decision_counts.get(trace.policy_name, 0) + len(trace.turns)
    fallback_rows: list[dict[str, float | int | str]] = []
    for policy_name in policy_names_for_fallback:
        stats = model_stats.get(policy_name, {}).get("inference_stats", {})
        parse_failures = int(stats.get("parse_failures", 0))
        fallback_calls = int(stats.get("fallback_calls", 0))
        total_decisions = int(stats.get("total_decisions", decision_counts.get(policy_name, 0)))
        fallback_rate = (
            float(stats.get("parse_failure_rate"))
            if "parse_failure_rate" in stats
            else (parse_failures / max(1, total_decisions))
        )
        fallback_rows.append(
            {
                "policy": policy_name,
                "total_decisions": total_decisions,
                "parse_failures": parse_failures,
                "fallback_calls": fallback_calls,
                "fallback_rate": fallback_rate,
            }
        )
    fallback_path = output_dir / "fallback_metrics.csv"
    with fallback_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "policy",
                "total_decisions",
                "parse_failures",
                "fallback_calls",
                "fallback_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(fallback_rows)

    fallback_gate_error: Exception | None = None
    if max_fallback_rate is not None:
        try:
            _validate_fallback_rates(fallback_rows, max_fallback_rate)
        except Exception as exc:
            fallback_gate_error = exc

    write_json(
        output_dir / "run_meta.json",
        {
            "runner": "checkpoint_eval",
            "split": split,
            "scenarios": len(scenarios),
            "repeats": repeats,
            "seed": seed,
            "domains": sorted(domains) if domains is not None else "all",
            "include_delay_mechanisms": (
                sorted(include_delay_mechanisms) if include_delay_mechanisms is not None else "all"
            ),
            "exclude_delay_mechanisms": (
                sorted(exclude_delay_mechanisms) if exclude_delay_mechanisms is not None else []
            ),
            "policies": policy_names_for_fallback,
            "model_policies": model_stats,
            "fallback_metrics_path": str(fallback_path),
            "max_fallback_rate": max_fallback_rate,
            "fallback_gate_passed": fallback_gate_error is None,
            "fallback_gate_error": str(fallback_gate_error) if fallback_gate_error is not None else None,
            "traces": len(traces),
            "records": len(records),
        },
    )
    if fallback_gate_error is not None:
        raise fallback_gate_error
    print(f"Wrote {len(traces)} traces and {len(records)} records to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=Path("artifacts/data/variant_tasks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model_eval"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenarios", type=int, default=1200)
    parser.add_argument(
        "--model-policy",
        action="append",
        default=[],
        help="Repeatable. Format: policy_name=/path/to/checkpoint",
    )
    parser.add_argument(
        "--include-baselines",
        type=str,
        default="runtime_verification_only",
        help="Comma-separated baseline policy names to include alongside checkpoint policies.",
    )
    parser.add_argument(
        "--fallback-policy",
        type=str,
        default="runtime_verification_only",
        help="Fallback policy used when model output cannot be parsed.",
    )
    parser.add_argument("--domains", type=str, default="")
    parser.add_argument("--include-delay-mechanisms", type=str, default="")
    parser.add_argument("--exclude-delay-mechanisms", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--training-traces", type=Path, default=None,
                        help="Path to training episode_traces.jsonl for contamination check.")
    parser.add_argument("--max-fallback-rate", type=float, default=None,
                        help="Fail if any policy fallback rate exceeds this threshold.")
    parser.add_argument("--sampling-seed", type=int, default=None,
                        help="Seed for scenario sampling (default: use --seed).")
    args = parser.parse_args()
    domains = {d.strip() for d in args.domains.split(",") if d.strip()} or None
    include_delay_mechanisms = (
        {m.strip() for m in args.include_delay_mechanisms.split(",") if m.strip()} or None
    )
    exclude_delay_mechanisms = (
        {m.strip() for m in args.exclude_delay_mechanisms.split(",") if m.strip()} or None
    )
    baseline_names = [name.strip() for name in args.include_baselines.split(",") if name.strip()]
    run(
        variants=args.variants,
        output_dir=args.output_dir,
        split=args.split,
        repeats=args.repeats,
        seed=args.seed,
        max_scenarios=args.max_scenarios,
        model_policies=args.model_policy,
        include_baselines=baseline_names,
        fallback_policy_name=args.fallback_policy,
        domains=domains,
        include_delay_mechanisms=include_delay_mechanisms,
        exclude_delay_mechanisms=exclude_delay_mechanisms,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        training_traces=args.training_traces,
        max_fallback_rate=args.max_fallback_rate,
        sampling_seed=args.sampling_seed,
    )


if __name__ == "__main__":
    main()

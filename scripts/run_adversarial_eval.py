from __future__ import annotations

import argparse
import gc
from pathlib import Path

from defer.baselines.runner import RunnerConfig, run_policies
from defer.baselines.policies import policy_registry
from defer.core.io import write_json, write_jsonl
from defer.sim.adversarial_scenarios import (
    AdversarialScenarioConfig,
    generate_adversarial_scenarios,
)

import csv


def _release_model_policy(policy) -> None:
    for attr in ("model", "tokenizer"):
        if hasattr(policy, attr):
            delattr(policy, attr)
    gc.collect()
    try:
        import torch
    except Exception:  # pragma: no cover
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(
    output_dir: Path,
    repeats: int,
    seed: int,
    max_scenarios: int,
    include_baselines: list[str],
    model_policies: dict[str, str] | None = None,
) -> None:
    config = AdversarialScenarioConfig(n_scenarios=max_scenarios, seed=seed)
    scenarios = generate_adversarial_scenarios(config)

    registry = policy_registry()
    policy_names: list[str] = []
    traces = []
    records = []
    for baseline_name in include_baselines:
        if baseline_name in registry:
            policy_names.append(baseline_name)

    baseline_policies = [registry[name] for name in policy_names]
    if baseline_policies:
        baseline_traces, baseline_records = run_policies(
            scenarios=scenarios,
            policies=baseline_policies,
            config=RunnerConfig(repeats=repeats, seed=seed),
        )
        traces.extend(baseline_traces)
        records.extend(baseline_records)

    model_stats: dict[str, dict] = {}
    if model_policies:
        from defer.baselines.model_policy import HFCheckpointPolicy

        for name, path in model_policies.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Checkpoint path for policy '{name}' does not exist: {path}")
            fallback = registry.get("runtime_verification_only", list(registry.values())[0])
            policy_names.append(name)
            policy = HFCheckpointPolicy(name=name, checkpoint_path=path, fallback_policy=fallback)
            try:
                model_traces, model_records = run_policies(
                    scenarios=scenarios,
                    policies=[policy],
                    config=RunnerConfig(repeats=repeats, seed=seed),
                )
                traces.extend(model_traces)
                records.extend(model_records)
                model_stats[name] = policy.stats()
            finally:
                _release_model_policy(policy)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output_dir / "episode_traces.jsonl",
        [trace.model_dump(mode="json") for trace in traces],
    )
    write_jsonl(
        output_dir / "reliability_records.jsonl",
        [record.model_dump(mode="json") for record in records],
    )

    decision_counts: dict[str, int] = {}
    for trace in traces:
        decision_counts[trace.policy_name] = decision_counts.get(trace.policy_name, 0) + len(trace.turns)
    fallback_rows = []
    for policy_name in policy_names:
        stats = model_stats.get(policy_name, {})
        parse_failures = int(stats.get("parse_failures", 0))
        fallback_calls = int(stats.get("fallback_calls", 0))
        total_decisions = int(stats.get("total_decisions", decision_counts.get(policy_name, 0)))
        fallback_rate = (
            float(stats.get("parse_failure_rate"))
            if "parse_failure_rate" in stats
            else (parse_failures / max(1, total_decisions))
        )
        fallback_rows.append({
            "policy": policy_name,
            "total_decisions": total_decisions,
            "parse_failures": parse_failures,
            "fallback_calls": fallback_calls,
            "fallback_rate": fallback_rate,
        })
    fallback_path = output_dir / "fallback_metrics.csv"
    with fallback_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["policy", "total_decisions", "parse_failures", "fallback_calls", "fallback_rate"],
        )
        writer.writeheader()
        writer.writerows(fallback_rows)

    write_json(
        output_dir / "run_meta.json",
        {
            "runner": "adversarial_eval",
            "scenarios": len(scenarios),
            "repeats": repeats,
            "seed": seed,
            "policies": policy_names,
            "traces": len(traces),
            "records": len(records),
        },
    )
    print(f"Wrote {len(traces)} adversarial traces and {len(records)} records to {output_dir}")


def _parse_model_policies(items: list[str]) -> dict[str, str]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --model-policy '{item}'. Expected name=path")
        name, path = item.split("=", 1)
        result[name.strip()] = path.strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/adversarial_eval"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenarios", type=int, default=200)
    parser.add_argument(
        "--include-baselines",
        type=str,
        default="runtime_verification_only,react",
    )
    parser.add_argument(
        "--model-policy",
        action="append",
        default=[],
        help="Repeatable. Format: policy_name=model_path",
    )
    args = parser.parse_args()
    include_baselines = [x.strip() for x in args.include_baselines.split(",") if x.strip()]
    model_policies = _parse_model_policies(args.model_policy) if args.model_policy else None
    run(
        output_dir=args.output_dir,
        repeats=args.repeats,
        seed=args.seed,
        max_scenarios=args.max_scenarios,
        include_baselines=include_baselines,
        model_policies=model_policies,
    )


if __name__ == "__main__":
    main()

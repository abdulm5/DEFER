from __future__ import annotations

import argparse
import csv
from pathlib import Path

from defer.baselines.model_policy import HFCheckpointPolicy, InferenceConfig
from defer.baselines.policies import policy_registry
from defer.baselines.runner import RunnerConfig, run_policies
from defer.core.io import read_jsonl, write_json, write_jsonl
from defer.data.schema import VariantTask
from defer.sim.scenario import Scenario


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
        print(f"No scenarios found for split={split}; falling back to all splits.")
        scenarios = _rows_to_scenarios(
            rows,
            split=None,
            domains=domains,
            include_delay_mechanisms=include_delay_mechanisms,
            exclude_delay_mechanisms=exclude_delay_mechanisms,
        )
    scenarios = scenarios[:max_scenarios]

    registry = policy_registry()
    if fallback_policy_name not in registry:
        raise ValueError(f"Unknown fallback policy: {fallback_policy_name}")

    policies = []
    baseline_names = include_baselines or []
    for baseline_name in baseline_names:
        if baseline_name not in registry:
            raise ValueError(f"Unknown baseline policy in --include-baselines: {baseline_name}")
        policies.append(registry[baseline_name])

    model_stats: dict[str, dict] = {}
    for name, checkpoint_path in _parse_model_specs(model_policies):
        fallback = registry.get(name, registry[fallback_policy_name])
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
        policies.append(policy)
        model_stats[name] = {
            "checkpoint_path": str(checkpoint_path),
            "fallback_policy": fallback.name,
        }

    traces, records = run_policies(
        scenarios=scenarios,
        policies=policies,
        config=RunnerConfig(repeats=repeats, seed=seed),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "episode_traces.jsonl", [trace.model_dump(mode="json") for trace in traces])
    write_jsonl(
        output_dir / "reliability_records.jsonl",
        [record.model_dump(mode="json") for record in records],
    )

    for policy in policies:
        if isinstance(policy, HFCheckpointPolicy):
            model_stats[policy.name]["inference_stats"] = policy.stats()

    decision_counts: dict[str, int] = {}
    for trace in traces:
        decision_counts[trace.policy_name] = decision_counts.get(trace.policy_name, 0) + len(trace.turns)
    fallback_rows: list[dict[str, float | int | str]] = []
    for policy in policies:
        stats = model_stats.get(policy.name, {}).get("inference_stats", {})
        parse_failures = int(stats.get("parse_failures", 0))
        fallback_calls = int(stats.get("fallback_calls", 0))
        total_decisions = int(stats.get("total_decisions", decision_counts.get(policy.name, 0)))
        fallback_rate = (
            float(stats.get("parse_failure_rate"))
            if "parse_failure_rate" in stats
            else (parse_failures / max(1, total_decisions))
        )
        fallback_rows.append(
            {
                "policy": policy.name,
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
            "policies": [policy.name for policy in policies],
            "model_policies": model_stats,
            "fallback_metrics_path": str(fallback_path),
            "traces": len(traces),
            "records": len(records),
        },
    )
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
    )


if __name__ == "__main__":
    main()

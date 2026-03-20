from __future__ import annotations

import argparse
from pathlib import Path

from defer.baselines.runner import RunnerConfig, run_policies
from defer.baselines.policies import policy_registry
from defer.core.io import write_json, write_jsonl
from defer.sim.adversarial_scenarios import (
    AdversarialScenarioConfig,
    generate_adversarial_scenarios,
)

import csv


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
    policies = []
    for baseline_name in include_baselines:
        if baseline_name in registry:
            policies.append(registry[baseline_name])

    if model_policies:
        from defer.baselines.model_policy import HFCheckpointPolicy

        for name, path in model_policies.items():
            fallback = registry.get("runtime_verification_only", list(registry.values())[0])
            policies.append(HFCheckpointPolicy(name=name, checkpoint_path=path, fallback_policy=fallback))

    traces, records = run_policies(
        scenarios=scenarios,
        policies=policies,
        config=RunnerConfig(repeats=repeats, seed=seed),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output_dir / "episode_traces.jsonl",
        [trace.model_dump(mode="json") for trace in traces],
    )
    write_jsonl(
        output_dir / "reliability_records.jsonl",
        [record.model_dump(mode="json") for record in records],
    )

    fallback_rows = []
    for policy in policies:
        fallback_rows.append(
            {
                "policy": policy.name,
                "total_decisions": sum(
                    len(t.turns) for t in traces if t.policy_name == policy.name
                ),
                "parse_failures": 0,
                "fallback_calls": 0,
                "fallback_rate": 0.0,
            }
        )
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
            "policies": [p.name for p in policies],
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

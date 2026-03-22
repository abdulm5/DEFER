from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from defer.core.io import read_jsonl, write_json, write_jsonl
from scripts.run_checkpoint_eval import run as run_checkpoint_eval_run

_DEFAULT_EVAL_SEED = 42


def _format_policy_templates(
    templates: list[str],
    seed: int,
    allow_static_model_policy_templates: bool = False,
) -> list[str]:
    specs: list[str] = []
    for template in templates:
        if "=" not in template:
            raise ValueError(
                f"Invalid --model-policy-template '{template}'. "
                "Expected format policy_name=/path/with/{seed}/placeholder"
            )
        name, path_template = template.split("=", 1)
        path_template = path_template.strip()
        has_seed_placeholder = "{seed}" in path_template
        if (not has_seed_placeholder) and (not allow_static_model_policy_templates):
            raise ValueError(
                f"Model policy template for '{name.strip()}' must include '{{seed}}' "
                "to avoid silently reusing the same checkpoint across all sweep seeds. "
                "Pass --allow-static-model-policy-templates to override."
            )
        path = path_template.format(seed=seed) if has_seed_placeholder else path_template
        specs.append(f"{name.strip()}={path}")
    return specs


def run(
    variants: Path,
    output_dir: Path,
    seeds_config: Path,
    repeats: int,
    split: str,
    max_scenarios: int,
    model_policy_templates: list[str],
    include_baselines: str,
    fallback_policy: str,
    domains: set[str] | None = None,
    include_delay_mechanisms: set[str] | None = None,
    exclude_delay_mechanisms: set[str] | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    allow_static_model_policy_templates: bool = False,
) -> None:
    payload = json.loads(seeds_config.read_text(encoding="utf-8"))
    seeds = list(payload.get("primary_model_seeds", [])) + list(
        payload.get("confirmatory_model_seeds", [])
    )
    eval_seed = int(payload.get("eval_seed", _DEFAULT_EVAL_SEED))
    if not seeds:
        raise ValueError(f"No seeds found in {seeds_config}")
    if not model_policy_templates:
        raise ValueError("At least one --model-policy-template is required.")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_records: list[dict] = []
    merged_traces: list[dict] = []
    fallback_rows_by_seed: list[dict] = []

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        run_checkpoint_eval_run(
            variants=variants,
            output_dir=seed_dir,
            split=split,
            repeats=repeats,
            seed=seed,
            max_scenarios=max_scenarios,
            model_policies=_format_policy_templates(
                model_policy_templates,
                seed=seed,
                allow_static_model_policy_templates=allow_static_model_policy_templates,
            ),
            include_baselines=[x.strip() for x in include_baselines.split(",") if x.strip()],
            fallback_policy_name=fallback_policy,
            domains=domains,
            include_delay_mechanisms=include_delay_mechanisms,
            exclude_delay_mechanisms=exclude_delay_mechanisms,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            sampling_seed=eval_seed,
        )
        merged_records.extend(read_jsonl(seed_dir / "reliability_records.jsonl"))
        merged_traces.extend(read_jsonl(seed_dir / "episode_traces.jsonl"))
        fallback_path = seed_dir / "fallback_metrics.csv"
        if fallback_path.exists():
            with fallback_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    fallback_rows_by_seed.append(
                        {
                            "seed": seed,
                            "policy": row.get("policy", ""),
                            "total_decisions": int(row.get("total_decisions", 0) or 0),
                            "parse_failures": int(row.get("parse_failures", 0) or 0),
                            "fallback_calls": int(row.get("fallback_calls", 0) or 0),
                            "fallback_rate": float(row.get("fallback_rate", 0.0) or 0.0),
                        }
                    )

    write_jsonl(output_dir / "merged_reliability_records.jsonl", merged_records)
    write_jsonl(output_dir / "merged_episode_traces.jsonl", merged_traces)

    fallback_path = output_dir / "fallback_metrics.csv"
    fallback_seed_path = output_dir / "fallback_metrics_by_seed.csv"
    if fallback_rows_by_seed:
        totals: dict[str, dict[str, int]] = {}
        for row in fallback_rows_by_seed:
            policy = row["policy"]
            bucket = totals.setdefault(
                policy,
                {"total_decisions": 0, "parse_failures": 0, "fallback_calls": 0},
            )
            bucket["total_decisions"] += int(row["total_decisions"])
            bucket["parse_failures"] += int(row["parse_failures"])
            bucket["fallback_calls"] += int(row["fallback_calls"])
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
            for policy, bucket in sorted(totals.items()):
                total_decisions = bucket["total_decisions"]
                parse_failures = bucket["parse_failures"]
                fallback_calls = bucket["fallback_calls"]
                writer.writerow(
                    {
                        "policy": policy,
                        "total_decisions": total_decisions,
                        "parse_failures": parse_failures,
                        "fallback_calls": fallback_calls,
                        "fallback_rate": parse_failures / max(1, total_decisions),
                    }
                )
        with fallback_seed_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "seed",
                    "policy",
                    "total_decisions",
                    "parse_failures",
                    "fallback_calls",
                    "fallback_rate",
                ],
            )
            writer.writeheader()
            writer.writerows(fallback_rows_by_seed)

    write_json(
        output_dir / "sweep_meta.json",
        {
            "seeds": seeds,
            "repeats": repeats,
            "split": split,
            "model_policy_templates": model_policy_templates,
            "allow_static_model_policy_templates": allow_static_model_policy_templates,
            "merged_records": len(merged_records),
            "merged_traces": len(merged_traces),
            "fallback_metrics_path": str(fallback_path) if fallback_rows_by_seed else None,
        },
    )
    print(f"Completed checkpoint seed sweep for {len(seeds)} seeds at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=Path("artifacts/data/variant_tasks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/checkpoint_seed_sweep"))
    parser.add_argument("--seeds-config", type=Path, default=Path("defer/configs/seeds.json"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-scenarios", type=int, default=1200)
    parser.add_argument(
        "--model-policy-template",
        action="append",
        default=[],
        help="Repeatable. Format: policy_name=/path/to/checkpoint_seed_{seed}/model",
    )
    parser.add_argument(
        "--include-baselines",
        type=str,
        default="runtime_verification_only",
        help="Comma-separated baseline policy names to include alongside checkpoint policies.",
    )
    parser.add_argument("--fallback-policy", type=str, default="runtime_verification_only")
    parser.add_argument("--domains", type=str, default="")
    parser.add_argument("--include-delay-mechanisms", type=str, default="")
    parser.add_argument("--exclude-delay-mechanisms", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--allow-static-model-policy-templates",
        action="store_true",
        help="Allow templates without {seed}. This weakens model-seed sweep validity.",
    )
    args = parser.parse_args()
    domains = {d.strip() for d in args.domains.split(",") if d.strip()} or None
    include_delay_mechanisms = (
        {m.strip() for m in args.include_delay_mechanisms.split(",") if m.strip()} or None
    )
    exclude_delay_mechanisms = (
        {m.strip() for m in args.exclude_delay_mechanisms.split(",") if m.strip()} or None
    )
    run(
        variants=args.variants,
        output_dir=args.output_dir,
        seeds_config=args.seeds_config,
        repeats=args.repeats,
        split=args.split,
        max_scenarios=args.max_scenarios,
        model_policy_templates=args.model_policy_template,
        include_baselines=args.include_baselines,
        fallback_policy=args.fallback_policy,
        domains=domains,
        include_delay_mechanisms=include_delay_mechanisms,
        exclude_delay_mechanisms=exclude_delay_mechanisms,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        allow_static_model_policy_templates=args.allow_static_model_policy_templates,
    )


if __name__ == "__main__":
    main()

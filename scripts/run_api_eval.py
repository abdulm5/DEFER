from __future__ import annotations

import argparse
import csv
from pathlib import Path

from defer.baselines.api_policy import APIInferenceConfig, OpenAIChatPolicy
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


def _parse_api_specs(items: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --api-policy '{item}'. Expected format policy_name=model_identifier"
            )
        name, model = item.split("=", 1)
        name = name.strip()
        model = model.strip()
        if not name or not model:
            raise ValueError(f"Invalid --api-policy '{item}'")
        specs.append((name, model))
    return specs


def run(
    variants: Path,
    output_dir: Path,
    split: str,
    repeats: int,
    seed: int,
    max_scenarios: int,
    api_policies: list[str],
    include_baselines: list[str] | None = None,
    fallback_policy_name: str = "runtime_verification_only",
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str = "https://api.openai.com/v1/chat/completions",
    domains: set[str] | None = None,
    include_delay_mechanisms: set[str] | None = None,
    exclude_delay_mechanisms: set[str] | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout_seconds: int = 60,
    system_prompt_override: str | None = None,
) -> None:
    if not api_policies:
        raise ValueError("At least one --api-policy name=model_id is required.")
    rows = read_jsonl(variants)
    scenarios = _rows_to_scenarios(
        rows,
        split=split,
        domains=domains,
        include_delay_mechanisms=include_delay_mechanisms,
        exclude_delay_mechanisms=exclude_delay_mechanisms,
    )
    if not scenarios:
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
    for baseline_name in include_baselines or []:
        if baseline_name not in registry:
            raise ValueError(f"Unknown baseline policy: {baseline_name}")
        policies.append(registry[baseline_name])

    system_prompt_map: dict[str, str] = {}
    if system_prompt_override == "prompted_deferral":
        from defer.baselines.prompted_deferral_policy import PROMPTED_DEFERRAL_SYSTEM_PROMPT
        system_prompt_map["prompted_deferral"] = PROMPTED_DEFERRAL_SYSTEM_PROMPT

    policy_stats: dict[str, dict] = {}
    for name, model in _parse_api_specs(api_policies):
        fallback = registry.get(name, registry[fallback_policy_name])
        extra_kwargs: dict = {}
        if name in system_prompt_map:
            extra_kwargs["system_prompt"] = system_prompt_map[name]
        policy = OpenAIChatPolicy(
            name=name,
            model=model,
            fallback_policy=fallback,
            api_key_env=api_key_env,
            base_url=base_url,
            inference=APIInferenceConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout_seconds=timeout_seconds,
            ),
            **extra_kwargs,
        )
        policies.append(policy)
        policy_stats[name] = {"model": model, "fallback_policy": fallback.name}

    traces, records = run_policies(
        scenarios=scenarios,
        policies=policies,
        config=RunnerConfig(repeats=repeats, seed=seed),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "episode_traces.jsonl", [trace.model_dump(mode="json") for trace in traces])
    write_jsonl(output_dir / "reliability_records.jsonl", [record.model_dump(mode="json") for record in records])

    decision_counts: dict[str, int] = {}
    for trace in traces:
        decision_counts[trace.policy_name] = decision_counts.get(trace.policy_name, 0) + len(trace.turns)
    fallback_rows = []
    for policy in policies:
        stats = policy_stats.get(policy.name, {})
        if isinstance(policy, OpenAIChatPolicy):
            stats["inference_stats"] = policy.stats()
            policy_stats[policy.name] = stats
        inference = stats.get("inference_stats", {})
        parse_failures = int(inference.get("parse_failures", 0))
        fallback_calls = int(inference.get("fallback_calls", 0))
        total_decisions = int(inference.get("total_decisions", decision_counts.get(policy.name, 0)))
        fallback_rows.append(
            {
                "policy": policy.name,
                "total_decisions": total_decisions,
                "parse_failures": parse_failures,
                "fallback_calls": fallback_calls,
                "fallback_rate": parse_failures / max(1, total_decisions),
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
            "runner": "api_eval",
            "split": split,
            "scenarios": len(scenarios),
            "repeats": repeats,
            "seed": seed,
            "policies": [policy.name for policy in policies],
            "api_policies": policy_stats,
            "fallback_metrics_path": str(fallback_path),
            "traces": len(traces),
            "records": len(records),
        },
    )
    print(f"Wrote {len(traces)} traces and {len(records)} records to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=Path("artifacts/data/variant_tasks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/api_eval"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenarios", type=int, default=1200)
    parser.add_argument(
        "--api-policy",
        action="append",
        default=[],
        help="Repeatable. Format: policy_name=model_identifier",
    )
    parser.add_argument(
        "--include-baselines",
        type=str,
        default="runtime_verification_only",
        help="Comma-separated baseline policies to include with API policies.",
    )
    parser.add_argument("--fallback-policy", type=str, default="runtime_verification_only")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--base-url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--domains", type=str, default="")
    parser.add_argument("--include-delay-mechanisms", type=str, default="")
    parser.add_argument("--exclude-delay-mechanisms", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument(
        "--system-prompt-override",
        type=str,
        default=None,
        help="Override system prompt for matching policy name. E.g. 'prompted_deferral'.",
    )
    args = parser.parse_args()
    domains = {d.strip() for d in args.domains.split(",") if d.strip()} or None
    include_delay_mechanisms = (
        {m.strip() for m in args.include_delay_mechanisms.split(",") if m.strip()} or None
    )
    exclude_delay_mechanisms = (
        {m.strip() for m in args.exclude_delay_mechanisms.split(",") if m.strip()} or None
    )
    include_baselines = [x.strip() for x in args.include_baselines.split(",") if x.strip()]
    run(
        variants=args.variants,
        output_dir=args.output_dir,
        split=args.split,
        repeats=args.repeats,
        seed=args.seed,
        max_scenarios=args.max_scenarios,
        api_policies=args.api_policy,
        include_baselines=include_baselines,
        fallback_policy_name=args.fallback_policy,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        domains=domains,
        include_delay_mechanisms=include_delay_mechanisms,
        exclude_delay_mechanisms=exclude_delay_mechanisms,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_seconds=args.timeout_seconds,
        system_prompt_override=args.system_prompt_override,
    )


if __name__ == "__main__":
    main()

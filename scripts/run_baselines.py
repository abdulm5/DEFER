from __future__ import annotations

import argparse
from pathlib import Path

from defer.baselines.runner import RunnerConfig, run_baselines
from defer.configs.defaults import BASELINES
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


def run(
    variants: Path,
    output_dir: Path,
    split: str,
    repeats: int,
    seed: int,
    max_scenarios: int,
    domains: set[str] | None = None,
    include_delay_mechanisms: set[str] | None = None,
    exclude_delay_mechanisms: set[str] | None = None,
) -> None:
    rows = read_jsonl(variants)
    scenarios = _rows_to_scenarios(
        rows,
        split=split,
        domains=domains,
        include_delay_mechanisms=include_delay_mechanisms,
        exclude_delay_mechanisms=exclude_delay_mechanisms,
    )
    if not scenarios:
        print(
            f"No scenarios found for split={split}; falling back to all splits for deterministic smoke run."
        )
        scenarios = _rows_to_scenarios(
            rows,
            split=None,
            domains=domains,
            include_delay_mechanisms=include_delay_mechanisms,
            exclude_delay_mechanisms=exclude_delay_mechanisms,
        )
    scenarios = scenarios[:max_scenarios]
    traces, records = run_baselines(
        scenarios=scenarios,
        selected_policies=BASELINES,
        config=RunnerConfig(repeats=repeats, seed=seed),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_rows = [trace.model_dump(mode="json") for trace in traces]
    record_rows = [record.model_dump(mode="json") for record in records]
    write_jsonl(output_dir / "episode_traces.jsonl", trace_rows)
    write_jsonl(output_dir / "reliability_records.jsonl", record_rows)
    write_json(
        output_dir / "run_meta.json",
        {
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
            "policies": BASELINES,
            "traces": len(trace_rows),
            "records": len(record_rows),
        },
    )
    print(f"Wrote {len(trace_rows)} traces and {len(record_rows)} records to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=Path("artifacts/data/variant_tasks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenarios", type=int, default=1000)
    parser.add_argument(
        "--domains",
        type=str,
        default="",
        help="Comma-separated domain filter (calendar,email,rest,sql).",
    )
    parser.add_argument(
        "--include-delay-mechanisms",
        type=str,
        default="",
        help="Comma-separated delay mechanism allowlist.",
    )
    parser.add_argument(
        "--exclude-delay-mechanisms",
        type=str,
        default="",
        help="Comma-separated delay mechanism denylist.",
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
        split=args.split,
        repeats=args.repeats,
        seed=args.seed,
        max_scenarios=args.max_scenarios,
        domains=domains,
        include_delay_mechanisms=include_delay_mechanisms,
        exclude_delay_mechanisms=exclude_delay_mechanisms,
    )


if __name__ == "__main__":
    main()

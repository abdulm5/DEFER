from __future__ import annotations

import argparse
import json
from pathlib import Path

from defer.core.io import read_jsonl, write_json, write_jsonl
from scripts.run_baselines import run as run_baselines_run


def run(
    variants: Path,
    output_dir: Path,
    seeds_config: Path,
    repeats: int,
    split: str,
    max_scenarios: int,
    domains: set[str] | None = None,
    include_delay_mechanisms: set[str] | None = None,
    exclude_delay_mechanisms: set[str] | None = None,
) -> None:
    seeds_payload = json.loads(seeds_config.read_text(encoding="utf-8"))
    seeds = list(seeds_payload.get("primary_model_seeds", [])) + list(
        seeds_payload.get("confirmatory_model_seeds", [])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_records: list[dict] = []
    merged_traces: list[dict] = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        run_baselines_run(
            variants=variants,
            output_dir=seed_dir,
            split=split,
            repeats=repeats,
            seed=seed,
            max_scenarios=max_scenarios,
            domains=domains,
            include_delay_mechanisms=include_delay_mechanisms,
            exclude_delay_mechanisms=exclude_delay_mechanisms,
        )
        merged_records.extend(read_jsonl(seed_dir / "reliability_records.jsonl"))
        merged_traces.extend(read_jsonl(seed_dir / "episode_traces.jsonl"))

    write_jsonl(output_dir / "merged_reliability_records.jsonl", merged_records)
    write_jsonl(output_dir / "merged_episode_traces.jsonl", merged_traces)
    write_json(
        output_dir / "sweep_meta.json",
        {"seeds": seeds, "repeats": repeats, "split": split, "merged_records": len(merged_records)},
    )
    print(f"Completed seed sweep for {len(seeds)} seeds at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=Path("artifacts/data/variant_tasks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/seed_sweep"))
    parser.add_argument("--seeds-config", type=Path, default=Path("defer/configs/seeds.json"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-scenarios", type=int, default=1200)
    parser.add_argument("--domains", type=str, default="")
    parser.add_argument("--include-delay-mechanisms", type=str, default="")
    parser.add_argument("--exclude-delay-mechanisms", type=str, default="")
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
        domains=domains,
        include_delay_mechanisms=include_delay_mechanisms,
        exclude_delay_mechanisms=exclude_delay_mechanisms,
    )


if __name__ == "__main__":
    main()

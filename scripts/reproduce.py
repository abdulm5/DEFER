from __future__ import annotations

import argparse
from pathlib import Path

from defer.core.io import write_json, write_jsonl
from defer.data.seeds import as_json_rows as seed_rows, generate_seed_tasks
from defer.data.schema import SeedTask
from defer.data.variants import as_json_rows as variant_rows, generate_variants
from scripts.evaluate_metrics import run as evaluate_run
from scripts.run_baselines import run as run_baselines_run

def run(
    output_root: Path,
    tasks_per_domain: int,
    repeats: int,
    seed: int,
    max_scenarios: int,
) -> None:
    data_dir = output_root / "data"
    runs_dir = output_root / "runs"
    metrics_dir = output_root / "metrics"
    tables_dir = output_root / "tables"
    for path in [data_dir, runs_dir, metrics_dir, tables_dir]:
        path.mkdir(parents=True, exist_ok=True)

    seeds = generate_seed_tasks(tasks_per_domain=tasks_per_domain, seed=seed)
    seed_path = data_dir / "seed_tasks.jsonl"
    write_jsonl(seed_path, seed_rows(seeds))

    variants = generate_variants([SeedTask(**row) for row in seed_rows(seeds)], seed=seed)
    variant_path = data_dir / "variant_tasks.jsonl"
    write_jsonl(variant_path, variant_rows(variants))

    run_baselines_run(
        variants=variant_path,
        output_dir=runs_dir,
        split="test",
        repeats=repeats,
        seed=seed,
        max_scenarios=max_scenarios,
    )
    evaluate_run(
        records_path=runs_dir / "reliability_records.jsonl",
        output_dir=metrics_dir,
        bootstrap_resamples=10_000,
        seed=seed,
    )

    # Copy key tables for paper-ready use.
    summary_src = metrics_dir / "summary_metrics.csv"
    ci_src = metrics_dir / "bootstrap_ci.csv"
    tables_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.joinpath("table_main_metrics.csv").write_text(summary_src.read_text(encoding="utf-8"), encoding="utf-8")
    tables_dir.joinpath("table_bootstrap_ci.csv").write_text(ci_src.read_text(encoding="utf-8"), encoding="utf-8")

    write_json(
        output_root / "repro_manifest.json",
        {
            "tasks_per_domain": tasks_per_domain,
            "repeats": repeats,
            "seed": seed,
            "paths": {
                "seed_tasks": str(seed_path),
                "variant_tasks": str(variant_path),
                "runs": str(runs_dir),
                "metrics": str(metrics_dir),
                "tables": str(tables_dir),
            },
        },
    )
    print(f"Repro pipeline complete under {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--tasks-per-domain", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenarios", type=int, default=1200)
    args = parser.parse_args()
    run(
        output_root=args.output_root,
        tasks_per_domain=args.tasks_per_domain,
        repeats=args.repeats,
        seed=args.seed,
        max_scenarios=args.max_scenarios,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from defer.core.io import write_json, write_jsonl
from defer.data.seeds import as_json_rows, generate_seed_tasks

def run(
    output: Path,
    tasks_per_domain: int,
    seed: int,
) -> None:
    tasks = generate_seed_tasks(tasks_per_domain=tasks_per_domain, seed=seed)
    rows = as_json_rows(tasks)
    write_jsonl(output, rows)
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        split_counts[row["split"]] += 1
    write_json(output.with_suffix(".meta.json"), {"count": len(rows), "split_counts": split_counts, "seed": seed})
    print(f"Wrote {len(rows)} seed tasks to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/data/seed_tasks.jsonl"))
    parser.add_argument("--tasks-per-domain", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(output=args.output, tasks_per_domain=args.tasks_per_domain, seed=args.seed)


if __name__ == "__main__":
    main()

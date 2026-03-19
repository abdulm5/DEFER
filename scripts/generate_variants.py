from __future__ import annotations

import argparse
from pathlib import Path

from defer.core.io import read_jsonl, write_json, write_jsonl
from defer.data.schema import SeedTask
from defer.data.variants import as_json_rows, generate_variants

def run(
    seed_tasks: Path,
    output: Path,
    seed: int,
) -> None:
    rows = read_jsonl(seed_tasks)
    seeds = [SeedTask(**row) for row in rows]
    variants = generate_variants(seed_tasks=seeds, seed=seed)
    out_rows = as_json_rows(variants)
    write_jsonl(output, out_rows)

    write_json(
        output.with_suffix(".meta.json"),
        {
            "count": len(out_rows),
            "seed": seed,
            "epsilon_grid": [0.0, 0.1, 0.2, 0.3],
            "lambda_grid": [0.0, 0.1, 0.2, 0.3],
            "delay_settings": ["immediate", "delayed"],
        },
    )
    print(f"Wrote {len(out_rows)} variants to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-tasks", type=Path, default=Path("artifacts/data/seed_tasks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data/variant_tasks.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(seed_tasks=args.seed_tasks, output=args.output, seed=args.seed)


if __name__ == "__main__":
    main()

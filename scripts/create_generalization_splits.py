from __future__ import annotations

import argparse
from pathlib import Path

from defer.core.io import read_jsonl, write_json, write_jsonl


def run(
    variants_path: Path,
    output_dir: Path,
    heldout_delay_mechanisms: list[str],
) -> None:
    rows = read_jsonl(variants_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    delay_train = [
        row
        for row in rows
        if row.get("delay_mechanism", "none") not in set(heldout_delay_mechanisms)
    ]
    delay_eval = [
        row
        for row in rows
        if row.get("delay_mechanism", "none") in set(heldout_delay_mechanisms)
    ]

    delay_dir = output_dir / "delay_holdout"
    delay_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(delay_dir / "train_variants.jsonl", delay_train)
    write_jsonl(delay_dir / "eval_variants.jsonl", delay_eval)

    domain_dir = output_dir / "domain_holdout"
    domain_dir.mkdir(parents=True, exist_ok=True)
    domains = sorted({row.get("domain", "unknown") for row in rows})
    domain_stats: dict[str, dict[str, int]] = {}
    for heldout_domain in domains:
        train_rows = [row for row in rows if row.get("domain") != heldout_domain]
        eval_rows = [row for row in rows if row.get("domain") == heldout_domain]
        write_jsonl(domain_dir / f"{heldout_domain}_train_variants.jsonl", train_rows)
        write_jsonl(domain_dir / f"{heldout_domain}_eval_variants.jsonl", eval_rows)
        domain_stats[heldout_domain] = {"train_rows": len(train_rows), "eval_rows": len(eval_rows)}

    manifest = {
        "variants_path": str(variants_path),
        "heldout_delay_mechanisms": heldout_delay_mechanisms,
        "delay_holdout": {
            "train_rows": len(delay_train),
            "eval_rows": len(delay_eval),
            "train_path": str(delay_dir / "train_variants.jsonl"),
            "eval_path": str(delay_dir / "eval_variants.jsonl"),
        },
        "domain_holdout": domain_stats,
    }
    write_json(output_dir / "generalization_splits_manifest.json", manifest)
    print(f"Wrote generalization splits under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants-path",
        type=Path,
        default=Path("artifacts/paper_run_v1/data/variant_tasks.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/paper_run_v1/generalization_splits"),
    )
    parser.add_argument(
        "--heldout-delay-mechanisms",
        type=str,
        default="stale_schema_cache,cross_tool_evidence_lag",
    )
    args = parser.parse_args()
    run(
        variants_path=args.variants_path,
        output_dir=args.output_dir,
        heldout_delay_mechanisms=[m.strip() for m in args.heldout_delay_mechanisms.split(",") if m.strip()],
    )


if __name__ == "__main__":
    main()

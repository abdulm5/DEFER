from __future__ import annotations

import argparse
from pathlib import Path

from defer.analysis.human_eval import (
    aggregate_annotations,
    compute_inter_annotator_agreement,
)
from defer.core.io import read_jsonl, write_json


def run(
    annotations_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> None:
    annotations = read_jsonl(annotations_path)
    agreement = compute_inter_annotator_agreement(annotations)
    aggregated = aggregate_annotations(annotations, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "agreement_scores.json", agreement)
    write_json(output_dir / "human_eval_results.json", aggregated)
    print(f"Wrote human eval results to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(
        annotations_path=args.annotations_path,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

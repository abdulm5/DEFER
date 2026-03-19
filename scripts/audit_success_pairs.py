from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd

from defer.core.io import read_jsonl, write_json, write_jsonl


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def run(
    pairs_path: Path,
    output_dir: Path,
    sample_size: int = 50,
    seed: int = 42,
) -> None:
    rows = read_jsonl(pairs_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        write_json(output_dir / "audit_summary.json", {"pair_count": 0, "reason": "no_pairs"})
        pd.DataFrame().to_csv(output_dir / "audit_pairs.csv", index=False)
        write_jsonl(output_dir / "manual_review_sample.jsonl", [])
        print(f"No pairs found at {pairs_path}")
        return

    frame = pd.DataFrame(rows)
    for col in [
        "chosen_success",
        "rejected_success",
        "chosen_corrupt_success",
        "rejected_corrupt_success",
        "chosen_invalid_commit",
        "rejected_invalid_commit",
        "chosen_turn_budget_exhausted",
        "rejected_turn_budget_exhausted",
        "chosen_premature_commit",
        "rejected_premature_commit",
        "chosen_unnecessary_deferral",
        "rejected_unnecessary_deferral",
    ]:
        if col in frame.columns:
            frame[col] = frame[col].map(_as_bool)
        else:
            frame[col] = False
    for col in [
        "chosen_commit_timing_score",
        "rejected_commit_timing_score",
        "chosen_over_deferral_rate",
        "rejected_over_deferral_rate",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        else:
            frame[col] = 0.0

    frame["success_margin"] = frame["chosen_success"].astype(int) - frame["rejected_success"].astype(int)
    frame["timing_margin"] = (
        frame["chosen_commit_timing_score"] - frame["rejected_commit_timing_score"]
    )
    frame["timing_aligned"] = frame["timing_margin"] > 0.0
    frame["timing_counterexample"] = frame["timing_margin"] <= 0.0
    frame["chosen_premature_vs_rejected_not"] = (
        frame["chosen_premature_commit"] & (~frame["rejected_premature_commit"])
    )
    frame["chosen_less_premature"] = (
        (~frame["chosen_premature_commit"]) & frame["rejected_premature_commit"]
    )

    pair_count = int(len(frame))
    summary = {
        "pair_count": pair_count,
        "chosen_success_rate": float(frame["chosen_success"].mean()),
        "rejected_success_rate": float(frame["rejected_success"].mean()),
        "chosen_corrupt_success_rate": float(frame["chosen_corrupt_success"].mean()),
        "rejected_corrupt_success_rate": float(frame["rejected_corrupt_success"].mean()),
        "chosen_premature_commit_rate": float(frame["chosen_premature_commit"].mean()),
        "rejected_premature_commit_rate": float(frame["rejected_premature_commit"].mean()),
        "chosen_unnecessary_deferral_rate": float(frame["chosen_unnecessary_deferral"].mean()),
        "rejected_unnecessary_deferral_rate": float(frame["rejected_unnecessary_deferral"].mean()),
        "avg_timing_margin_chosen_minus_rejected": float(frame["timing_margin"].mean()),
        "timing_aligned_fraction": float(frame["timing_aligned"].mean()),
        "timing_counterexample_fraction": float(frame["timing_counterexample"].mean()),
        "chosen_less_premature_fraction": float(frame["chosen_less_premature"].mean()),
        "chosen_more_premature_fraction": float(frame["chosen_premature_vs_rejected_not"].mean()),
        "signal_purity_note": (
            "Lower timing_aligned_fraction and higher timing_counterexample_fraction indicate "
            "the success-signal pairs are less entangled with commit-timing signal."
        ),
    }
    write_json(output_dir / "audit_summary.json", summary)

    audit_cols = [
        "scenario_id",
        "chosen_policy",
        "rejected_policy",
        "quality_margin",
        "chosen_success",
        "rejected_success",
        "chosen_corrupt_success",
        "rejected_corrupt_success",
        "chosen_premature_commit",
        "rejected_premature_commit",
        "chosen_unnecessary_deferral",
        "rejected_unnecessary_deferral",
        "chosen_over_deferral_rate",
        "rejected_over_deferral_rate",
        "chosen_commit_timing_score",
        "rejected_commit_timing_score",
        "success_margin",
        "timing_margin",
        "timing_aligned",
        "timing_counterexample",
    ]
    frame[audit_cols].to_csv(output_dir / "audit_pairs.csv", index=False)

    rng = random.Random(seed)
    manual_pool = rows.copy()
    rng.shuffle(manual_pool)
    manual = manual_pool[:sample_size]
    write_jsonl(output_dir / "manual_review_sample.jsonl", manual)
    print(f"Wrote success-pair audit under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs-path",
        type=Path,
        default=Path("artifacts/data/dpo_pairs_success_signal.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/audits/success_pairs"),
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(
        pairs_path=args.pairs_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

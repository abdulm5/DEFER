from __future__ import annotations

import argparse
from pathlib import Path

from defer.training.dpo import build_dpo_manifest, run_dpo_training

def run(
    output_dir: Path,
    model_name: str,
    train_pairs: str,
    val_pairs: str,
    seed: int,
    mode: str,
    dry_run: bool,
) -> None:
    manifest = build_dpo_manifest(
        output_dir=output_dir,
        model_name=model_name,
        pair_train_path=train_pairs,
        pair_val_path=val_pairs,
        seed=seed,
        mode=mode,
    )
    print(f"Wrote {mode.upper()} manifest to {output_dir}")
    if dry_run:
        print("Dry-run enabled. Install train extras and add DPO backend for execution.")
        return
    summary = run_dpo_training(manifest=manifest, output_dir=output_dir)
    print(f"{mode.upper()} training complete. Summary: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training/dpo"))
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--train-pairs", type=str, default="artifacts/data/dpo_train.jsonl")
    parser.add_argument("--val-pairs", type=str, default="artifacts/data/dpo_val.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="dpo")
    parser.set_defaults(dry_run=True)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--execute", dest="dry_run", action="store_false")
    args = parser.parse_args()
    run(
        output_dir=args.output_dir,
        model_name=args.model_name,
        train_pairs=args.train_pairs,
        val_pairs=args.val_pairs,
        seed=args.seed,
        mode=args.mode,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

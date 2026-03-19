from __future__ import annotations

import argparse
from pathlib import Path

from defer.training.sft import build_sft_manifest, run_sft_training

def run(
    output_dir: Path,
    model_name: str,
    train_path: str,
    val_path: str,
    seed: int,
    dry_run: bool,
) -> None:
    manifest = build_sft_manifest(
        output_dir=output_dir, model_name=model_name, train_path=train_path, val_path=val_path, seed=seed
    )
    print(f"Wrote SFT manifest to {output_dir}")
    if dry_run:
        print("Dry-run enabled. Install train extras and add trainer backend for execution.")
        return
    summary = run_sft_training(manifest=manifest, output_dir=output_dir)
    print(f"SFT training complete. Summary: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training/sft"))
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--train-path", type=str, default="artifacts/data/sft_train.jsonl")
    parser.add_argument("--val-path", type=str, default="artifacts/data/sft_val.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(dry_run=True)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--execute", dest="dry_run", action="store_false")
    args = parser.parse_args()
    run(
        output_dir=args.output_dir,
        model_name=args.model_name,
        train_path=args.train_path,
        val_path=args.val_path,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

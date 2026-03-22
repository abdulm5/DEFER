from __future__ import annotations

import argparse
from pathlib import Path

from defer.core.io import write_json
from scripts.build_preference_pairs import run as build_pairs_run
from scripts.build_training_datasets import run as build_training_data_run
from scripts.create_generalization_splits import run as create_generalization_splits_run
from scripts.evaluate_metrics import run as evaluate_run
from scripts.generate_seed_tasks import run as generate_seeds_run
from scripts.generate_variants import run as generate_variants_run
from scripts.run_baselines import run as run_baselines_run
from scripts.run_seed_sweep import run as run_seed_sweep_run


def run(
    output_root: Path,
    tasks_per_domain: int,
    baseline_repeats: int,
    seed: int,
    max_scenarios: int,
    baseline_bootstrap_resamples: int,
    sweep_repeats: int,
    sweep_bootstrap_resamples: int,
    min_episodes_per_cell: int,
    strict_coverage: bool,
    data_split: str,
    eval_split: str,
    seeds_config: Path,
    skip_seed_sweep: bool,
) -> None:
    """
    One-shot paper data collection orchestration:
    1) seed tasks + variants
    2) generalization split manifests
    3) baseline benchmark runs on data split (for training data synthesis)
    4) baseline benchmark runs on eval split (for baseline metrics)
    5) baseline metrics
    6) DEFER-targeted preference pairs
    7) training datasets
    8) multi-seed sweep + final metrics on eval split
    """
    data_dir = output_root / "data"
    baseline_runs_train_dir = output_root / "baseline_runs_train"
    baseline_runs_eval_dir = output_root / "baseline_runs_eval"
    baseline_metrics_dir = output_root / "baseline_metrics"
    seed_sweep_dir = output_root / "seed_sweep"
    seed_sweep_metrics_dir = output_root / "seed_sweep_metrics"

    for path in [data_dir, baseline_runs_train_dir, baseline_metrics_dir]:
        path.mkdir(parents=True, exist_ok=True)
    if eval_split != data_split:
        baseline_runs_eval_dir.mkdir(parents=True, exist_ok=True)
    if not skip_seed_sweep:
        seed_sweep_dir.mkdir(parents=True, exist_ok=True)
        seed_sweep_metrics_dir.mkdir(parents=True, exist_ok=True)
    strict_flag = " --strict-coverage" if strict_coverage else ""

    seed_tasks_path = data_dir / "seed_tasks.jsonl"
    variants_path = data_dir / "variant_tasks.jsonl"
    dpo_pairs_path = data_dir / "dpo_pairs.jsonl"

    print("[1/9] Generating seed tasks...")
    generate_seeds_run(output=seed_tasks_path, tasks_per_domain=tasks_per_domain, seed=seed)

    print("[2/9] Generating variants...")
    generate_variants_run(seed_tasks=seed_tasks_path, output=variants_path, seed=seed)

    print("[3/9] Creating generalization split manifests...")
    create_generalization_splits_run(
        variants_path=variants_path,
        output_dir=data_dir / "generalization_splits",
        heldout_delay_mechanisms=["stale_schema_cache", "cross_tool_evidence_lag"],
    )

    print("[4/9] Running baseline benchmark episodes for data split...")
    run_baselines_run(
        variants=variants_path,
        output_dir=baseline_runs_train_dir,
        split=data_split,
        repeats=baseline_repeats,
        seed=seed,
        max_scenarios=max_scenarios,
    )

    if eval_split != data_split:
        print("[5/9] Running baseline benchmark episodes for eval split...")
        run_baselines_run(
            variants=variants_path,
            output_dir=baseline_runs_eval_dir,
            split=eval_split,
            repeats=baseline_repeats,
            seed=seed,
            max_scenarios=max_scenarios,
        )
    else:
        baseline_runs_eval_dir = baseline_runs_train_dir

    print("[6/9] Evaluating baseline metrics...")
    evaluate_run(
        records_path=baseline_runs_eval_dir / "reliability_records.jsonl",
        output_dir=baseline_metrics_dir,
        bootstrap_resamples=baseline_bootstrap_resamples,
        seed=seed,
        min_episodes_per_cell=min_episodes_per_cell,
        strict_coverage=strict_coverage,
    )

    print("[7/9] Building DEFER-targeted preference pairs...")
    build_pairs_run(
        traces_path=baseline_runs_train_dir / "episode_traces.jsonl",
        output=dpo_pairs_path,
        chosen_policies=["defer_full"],
        commit_chosen_policies=["defer_full", "perfect_verifier_posttrain", "clean_sft_only"],
        commit_quality_chosen_policies=[
            "defer_full",
            "perfect_verifier_posttrain",
            "clean_sft_only",
            "runtime_verification_only",
        ],
        rejected_policies=[
            "runtime_verification_only",
            "react",
            "clean_sft_only",
            "stress_training_no_contracts",
            "perfect_verifier_posttrain",
        ],
        allow_same_policy=False,
        allow_chosen_fallback=False,
        allow_commit_chosen_fallback=False,
        allow_commit_quality_chosen_fallback=False,
        include_commit_quality_pairs=True,
        target_commit_ratio=0.33,
        target_commit_quality_ratio=0.34,
        decision_window_turns=5,
        min_quality_margin=0.05,
    )

    print("[8/9] Building SFT/DPO training datasets...")
    build_training_data_run(
        traces_path=baseline_runs_train_dir / "episode_traces.jsonl",
        pairs_path=dpo_pairs_path,
        output_dir=data_dir / "training",
        seed=seed,
        val_ratio=0.1,
    )

    if not skip_seed_sweep:
        print("[9/9] Running 5+3 seed sweep and final metrics...")
        run_seed_sweep_run(
            variants=variants_path,
            output_dir=seed_sweep_dir,
            seeds_config=seeds_config,
            repeats=sweep_repeats,
            split=eval_split,
            max_scenarios=max_scenarios,
        )
        evaluate_run(
            records_path=seed_sweep_dir / "merged_reliability_records.jsonl",
            output_dir=seed_sweep_metrics_dir,
            bootstrap_resamples=sweep_bootstrap_resamples,
            seed=seed,
            min_episodes_per_cell=min_episodes_per_cell,
            strict_coverage=strict_coverage,
        )
    else:
        print("[9/9] Seed sweep skipped by flag.")

    manual_commands = [
        f"python -m scripts.generate_seed_tasks --output {seed_tasks_path} --tasks-per-domain {tasks_per_domain} --seed {seed}",
        f"python -m scripts.generate_variants --seed-tasks {seed_tasks_path} --output {variants_path} --seed {seed}",
        (
            f"python -m scripts.create_generalization_splits --variants-path {variants_path} "
            f"--output-dir {data_dir / 'generalization_splits'} "
            "--heldout-delay-mechanisms stale_schema_cache,cross_tool_evidence_lag"
        ),
        (
            f"python -m scripts.run_baselines --variants {variants_path} --output-dir {baseline_runs_train_dir} "
            f"--split {data_split} --repeats {baseline_repeats} --seed {seed} --max-scenarios {max_scenarios}"
        ),
    ]
    if eval_split != data_split:
        manual_commands.append(
            (
                f"python -m scripts.run_baselines --variants {variants_path} --output-dir {baseline_runs_eval_dir} "
                f"--split {eval_split} --repeats {baseline_repeats} --seed {seed} --max-scenarios {max_scenarios}"
            )
        )
    manual_commands.extend(
        [
            (
                f"python -m scripts.evaluate_metrics --records-path {baseline_runs_eval_dir / 'reliability_records.jsonl'} "
                f"--output-dir {baseline_metrics_dir} --bootstrap-resamples {baseline_bootstrap_resamples} "
                f"--seed {seed} --min-episodes-per-cell {min_episodes_per_cell}{strict_flag}"
            ),
            (
                f"python -m scripts.build_preference_pairs --traces-path {baseline_runs_train_dir / 'episode_traces.jsonl'} "
                f"--output {dpo_pairs_path} --include-commit-quality-pairs --target-commit-ratio 0.33 "
                "--target-commit-quality-ratio 0.34 --decision-window-turns 5 "
                "--min-quality-margin 0.05 "
                "--commit-chosen-policies defer_full,perfect_verifier_posttrain,clean_sft_only "
                "--commit-quality-chosen-policies defer_full,perfect_verifier_posttrain,clean_sft_only,runtime_verification_only"
            ),
            (
                f"python -m scripts.build_training_datasets --traces-path {baseline_runs_train_dir / 'episode_traces.jsonl'} "
                f"--pairs-path {dpo_pairs_path} --output-dir {data_dir / 'training'} --seed {seed} --val-ratio 0.1"
            ),
            (
                f"python -m scripts.run_seed_sweep --variants {variants_path} --output-dir {seed_sweep_dir} "
                f"--seeds-config {seeds_config} --repeats {sweep_repeats} --split {eval_split} --max-scenarios {max_scenarios}"
            ),
            (
                f"python -m scripts.evaluate_metrics --records-path {seed_sweep_dir / 'merged_reliability_records.jsonl'} "
                f"--output-dir {seed_sweep_metrics_dir} --bootstrap-resamples {sweep_bootstrap_resamples} "
                f"--seed {seed} --min-episodes-per-cell {min_episodes_per_cell}{strict_flag}"
            ),
        ]
    )

    command_manifest = {
        "manual_commands": manual_commands,
        "params": {
            "tasks_per_domain": tasks_per_domain,
            "baseline_repeats": baseline_repeats,
            "seed": seed,
            "max_scenarios": max_scenarios,
            "data_split": data_split,
            "eval_split": eval_split,
            "baseline_bootstrap_resamples": baseline_bootstrap_resamples,
            "sweep_repeats": sweep_repeats,
            "sweep_bootstrap_resamples": sweep_bootstrap_resamples,
            "min_episodes_per_cell": min_episodes_per_cell,
            "strict_coverage": strict_coverage,
            "seeds_config": str(seeds_config),
            "skip_seed_sweep": skip_seed_sweep,
        },
        "paths": {
            "seed_tasks": str(seed_tasks_path),
            "variants": str(variants_path),
            "baseline_runs_train": str(baseline_runs_train_dir),
            "baseline_runs_eval": str(baseline_runs_eval_dir),
            "baseline_metrics": str(baseline_metrics_dir),
            "dpo_pairs": str(dpo_pairs_path),
            "training_data": str(data_dir / "training"),
            "generalization_splits": str(data_dir / "generalization_splits"),
            "seed_sweep": str(seed_sweep_dir),
            "seed_sweep_metrics": str(seed_sweep_metrics_dir),
        },
    }
    write_json(output_root / "paper_collection_manifest.json", command_manifest)
    print(f"Paper data collection complete under {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/paper_run_v1"))
    parser.add_argument("--tasks-per-domain", type=int, default=300)
    parser.add_argument("--baseline-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenarios", type=int, default=3000)
    parser.add_argument(
        "--data-split",
        type=str,
        default="train",
        help="Split used to generate training traces/pairs (default: train).",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="test",
        help="Split used for baseline and seed-sweep evaluation (default: test).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Deprecated alias for --data-split.",
    )
    parser.add_argument("--baseline-bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--sweep-repeats", type=int, default=5)
    parser.add_argument("--sweep-bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--min-episodes-per-cell", type=int, default=100)
    parser.add_argument("--strict-coverage", action="store_true")
    parser.add_argument("--seeds-config", type=Path, default=Path("defer/configs/seeds.json"))
    parser.add_argument("--skip-seed-sweep", action="store_true")
    args = parser.parse_args()
    data_split = args.data_split
    if args.split is not None:
        data_split = args.split
    run(
        output_root=args.output_root,
        tasks_per_domain=args.tasks_per_domain,
        baseline_repeats=args.baseline_repeats,
        seed=args.seed,
        max_scenarios=args.max_scenarios,
        baseline_bootstrap_resamples=args.baseline_bootstrap_resamples,
        sweep_repeats=args.sweep_repeats,
        sweep_bootstrap_resamples=args.sweep_bootstrap_resamples,
        min_episodes_per_cell=args.min_episodes_per_cell,
        strict_coverage=args.strict_coverage,
        data_split=data_split,
        eval_split=args.eval_split,
        seeds_config=args.seeds_config,
        skip_seed_sweep=args.skip_seed_sweep,
    )


if __name__ == "__main__":
    main()

# Failure Recovery Playbook

## Data generation failure

Symptoms:
- `seed_tasks.jsonl` or `variant_tasks.jsonl` missing/corrupt.

Recovery:
1. Re-run `python -m scripts.generate_seed_tasks --seed <seed>`.
2. Re-run `python -m scripts.generate_variants --seed <seed>`.
3. Validate with `python -c "from defer.core.io import read_jsonl; print(len(read_jsonl('...')))"`.

## Simulation run interruption

Symptoms:
- partial `episode_traces.jsonl`, missing `run_meta.json`.

Recovery:
1. Delete incomplete run directory.
2. Re-run `python -m scripts.run_baselines ...` with same seed.
3. Confirm deterministic record count: `#scenarios * #policies * repeats`.

## Metric computation crash

Symptoms:
- missing `summary_metrics.csv` or `bootstrap_ci.csv`.

Recovery:
1. Re-run `python -m scripts.evaluate_metrics`.
2. Reduce `--bootstrap-resamples` for debugging.
3. Re-run full resamples for final numbers.

## Training stage issues

Symptoms:
- missing manifests or failed trainer initialization.

Recovery:
1. Run `python -m scripts.build_training_datasets` to regenerate SFT/DPO datasets.
2. Run `python -m scripts.train_sft --dry-run`.
3. Run `python -m scripts.train_dpo --dry-run`.
4. Verify paths + install train extras (`pip install -e ".[train]"`).

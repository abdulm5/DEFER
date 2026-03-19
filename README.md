# DEFER

DEFER is a reproducible research scaffold for studying tool-using agents under:

- uncertain verification (stale/provisional verifier signals),
- delayed truth revelation (async and eventual-consistency effects),
- production-like perturbations and faults.

The repository implements:

- frozen interfaces (`ContractSpec`, `VerifierOutput`, `AgentAction`, `EpisodeTrace`),
- four enterprise-style domains (calendar, email, REST/API, SQL),
- deterministic event loop for delayed side effects,
- stress engine (perturbations + fault injection),
- baseline policies,
- evaluation metrics (`R(k, ε, λ)`, AURS, gated success, DCS with precision/recall components, IER, EFV),
- reproducibility scripts and engineering tests.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m scripts.reproduce --output-root artifacts/demo
pytest -q
```

## One-command pipeline

```bash
python -m scripts.reproduce \
  --output-root artifacts/full_run \
  --tasks-per-domain 300 \
  --repeats 5 \
  --seed 42

# Optional: run full 5+3 seed protocol from defer/configs/seeds.json
python -m scripts.run_seed_sweep \
  --variants artifacts/full_run/data/variant_tasks.jsonl \
  --output-dir artifacts/full_run/seed_sweep \
  --repeats 5

# One-shot paper-grade collection (baseline + pairs + seed sweep + metrics)
python -m scripts.run_paper_data_collection \
  --output-root artifacts/paper_run_v1 \
  --tasks-per-domain 300 \
  --baseline-repeats 3 \
  --seed 42 \
  --max-scenarios 2000 \
  --baseline-bootstrap-resamples 2000 \
  --sweep-repeats 5 \
  --sweep-bootstrap-resamples 10000
```

## Final End-to-End Script (Real Checkpoints)

Run the full training + checkpoint-eval + holdout suite with one command:

```bash
RUN_DIR=artifacts/final_run_v1 CLEAN_OLD=1 BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  ./scripts/run_final_data_collection.sh
```

Optional API baseline:

```bash
OPENAI_API_KEY=... RUN_API_BASELINE=1 ./scripts/run_final_data_collection.sh
```

OpenRouter example:

```bash
OPENROUTER_API_KEY=... \
RUN_API_BASELINE=1 \
API_KEY_ENV=OPENROUTER_API_KEY \
API_BASE_URL=https://openrouter.ai/api/v1/chat/completions \
API_MODEL=openai/gpt-4o \
API_POLICY_NAME=frontier_gpt4o_zero_shot \
./scripts/run_final_data_collection.sh
```

Paper run (Llama 3.1 8B):

```bash
RUN_DIR=artifacts/paper_llama31_8b_v1 \
CLEAN_OLD=1 \
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct \
./scripts/run_final_data_collection.sh
```

Artifacts are created under:

- `artifacts/<run>/data` (seed tasks + variants),
- `artifacts/<run>/runs` (traces + baseline episode records),
- `artifacts/<run>/metrics` (summary, CI, pairwise tests, coverage, domain/category/mechanism breakdowns),
- `artifacts/<run>/tables` (paper-ready metric tables).

## Training stubs

Three scripts support post-training:

- `python -m scripts.create_generalization_splits`
- `python -m scripts.build_training_datasets`
- `python -m scripts.train_sft --dry-run`
- `python -m scripts.train_dpo --dry-run`
- `python -m scripts.run_checkpoint_eval --help`
- `python -m scripts.run_checkpoint_seed_sweep --help`
- `python -m scripts.run_api_eval --help` (frontier zero-shot API baseline)

Preference-pair generation (DEFER-targeted by default):

- `python -m scripts.build_preference_pairs --traces-path <...> --output <...>`
- `python -m scripts.build_success_preference_pairs --traces-path <...> --output <...>` (success-signal DPO ablation)
- `python -m scripts.audit_success_pairs --pairs-path <...> --output-dir <...>` (purity audit before success-signal ablation)
- Defaults prefer `defer_full` as chosen and baseline failures as rejected.

They produce exact configs and command manifests for QLoRA SFT and DPO/IPO.
If `.[train]` dependencies are installed, `--execute` runs training directly.

Checkpoint evaluation bridge:

- `run_baselines` / `run_seed_sweep` evaluate proxy policies.
- `run_checkpoint_eval` evaluates real model checkpoints in-loop with the simulator.
- `run_checkpoint_seed_sweep` runs 5+3 seed sweeps for checkpoint policies via `{seed}` path templates.
- `run_api_eval` runs OpenAI-compatible API models in-loop (same action contract, fallback instrumentation).
- Checkpoint eval writes `fallback_metrics.csv`; metric tables surface `fallback_rate` for interpretation gating.

## Reproducibility commitments

- All random paths are seeded.
- Deterministic replay for event ordering and stress injection.
- Deterministic metric reducers.
- Engineering tests for parser/verifier/event-order/metric invariance/trace-to-table reproducibility.

Supporting docs:

- `docs/reproducibility.md`
- `docs/failure_recovery.md`
- `docs/hardware_profile.md`
- `docs/runtime_cost.md`

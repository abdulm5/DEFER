# DEFER: Calibrated Deferral Under Uncertain Verification and Delayed Truth Revelation

This repository contains the research code for **DEFER**, a reproducible benchmark and post-training pipeline for studying tool-using agents when verifier signals are uncertain and correctness is revealed only after delayed side effects.

DEFER is designed for paper-grade evaluation: the codebase includes a deterministic simulator, seeded data generation, baseline and model-policy evaluation, post-training data construction, checkpoint/API evaluation, and scripts that regenerate the main artifact tables.

## Abstract

Tool-using agents are often evaluated as if correctness is immediately observable and verifier signals are trustworthy. Real systems are less clean: tool outputs can be stale or provisional, asynchronous effects can resolve later, and an action that looks correct at execution time can be contradicted after the fact. DEFER studies this setting directly. The repository provides a controlled multi-domain environment with explicit defer/commit actions, delayed truth revelation, seeded perturbations and faults, and procedure-aware reliability metrics. It supports both baseline policies and post-trained model policies, with end-to-end scripts for reproducing paper results and generating analysis artifacts.

## What This Repository Includes

- A deterministic event-driven simulator for delayed truth revelation
- Tool-use domains spanning calendar, email, REST/API, SQL, and extended enterprise-style settings
- Explicit defer/commit actions such as waiting, refreshing, cross-checking, reversible commit, and irreversible commit
- Stress testing over prompt perturbation (`epsilon`) and tool fault intensity (`lambda`)
- Baseline policies plus evaluation paths for local checkpoints and OpenAI-compatible APIs
- Metrics for deferral quality and reliability, including `AURS`, `DCS`, `IER`, `EFV`, gated success, and related breakdowns
- Data-generation, seed-sweep, SFT, DPO, human-eval, and integrity-check scripts used by the paper workflow

## Setup

Create an environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For post-training and checkpoint-based experiments, install training extras as well:

```bash
pip install -e ".[train]"
```

## Quick Start

Smoke-test the repository with the minimal reproduction path:

```bash
python -m scripts.reproduce --output-root artifacts/demo
pytest -q
```

You can also use the small helper targets in the `Makefile`:

```bash
make setup
make test
make repro
```

## Reproducing Results

### 1. Minimal end-to-end reproduction

This generates seed tasks, runs baseline evaluation, computes metrics, and writes paper-style tables:

```bash
python -m scripts.reproduce \
  --output-root artifacts/repro_run \
  --tasks-per-domain 300 \
  --repeats 5 \
  --seed 42
```

### 2. Paper data collection pipeline

This is the main paper-oriented collection path for baseline evaluation, pair construction, seed sweeps, and metric aggregation:

```bash
python -m scripts.run_paper_data_collection \
  --output-root artifacts/paper_run_v1 \
  --tasks-per-domain 300 \
  --baseline-repeats 3 \
  --seed 42 \
  --max-scenarios 3000 \
  --baseline-bootstrap-resamples 2000 \
  --sweep-repeats 5 \
  --sweep-bootstrap-resamples 10000
```

Optional seed-sweep run from the configured 5+3 seed protocol:

```bash
python -m scripts.run_seed_sweep \
  --variants artifacts/paper_run_v1/data/variant_tasks.jsonl \
  --output-dir artifacts/paper_run_v1/seed_sweep \
  --repeats 5
```

### 3. Full training + checkpoint evaluation

For the full training/evaluation path with real checkpoints:

```bash
RUN_DIR=artifacts/final_run_v1 \
CLEAN_OLD=1 \
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
./scripts/run_final_data_collection.sh
```

Example with a different base model:

```bash
RUN_DIR=artifacts/paper_llama31_8b_v1 \
CLEAN_OLD=1 \
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct \
./scripts/run_final_data_collection.sh
```

Optional split override:

```bash
BASELINE_SPLIT=train \
EVAL_SPLIT=test \
./scripts/run_final_data_collection.sh
```

### 4. Optional API baselines

To include an OpenAI-compatible API baseline in the same collection flow:

```bash
OPENAI_API_KEY=... RUN_API_BASELINE=1 ./scripts/run_final_data_collection.sh
```

Azure Foundry / Azure OpenAI example:

```bash
AZURE_OPENAI_API_KEY=... \
RUN_API_BASELINE=1 \
API_KEY_ENV=AZURE_OPENAI_API_KEY \
API_AUTH_MODE=api_key \
API_KEY_HEADER=api-key \
API_BASE_URL=https://YOUR-RESOURCE.openai.azure.com/openai/v1/chat/completions \
API_QUERY_PARAM=api-version=2024-10-21 \
API_MODEL=gpt-4o \
API_POLICY_NAME=frontier_gpt4o_zero_shot \
./scripts/run_final_data_collection.sh
```

Supplementary multi-model API matrix:

```bash
python -m scripts.run_api_sota_matrix \
  --config defer/configs/api_sota_matrix.example.yaml \
  --output-root artifacts/final_run_v1/api_sota \
  --workers 2
```

## Training and Evaluation Scripts

The repository exposes the main paper pipeline through Python entry points and module scripts. Common commands include:

```bash
python -m scripts.create_generalization_splits
python -m scripts.build_training_datasets
python -m scripts.build_preference_pairs --traces-path <...> --output <...>
python -m scripts.build_success_preference_pairs --traces-path <...> --output <...>
python -m scripts.audit_success_pairs --pairs-path <...> --output-dir <...>
python -m scripts.train_sft --dry-run
python -m scripts.train_dpo --dry-run
python -m scripts.run_checkpoint_eval --help
python -m scripts.run_checkpoint_seed_sweep --help
python -m scripts.run_api_eval --help
python -m scripts.run_adversarial_eval --help
python -m scripts.prepare_human_eval --help
python -m scripts.analyze_human_eval --help
python -m scripts.check_paper_integrity --help
```

## Artifact Layout

Most runs write outputs under a single artifact root:

- `artifacts/<run>/data`: generated tasks, variants, and split metadata
- `artifacts/<run>/runs`: traces, episode-level records, and evaluation outputs
- `artifacts/<run>/metrics`: aggregate metrics, confidence intervals, pairwise tests, and breakdown tables
- `artifacts/<run>/tables`: paper-ready tables and summary exports

## Repository Structure

```text
defer/        Core package: simulator, domains, metrics, baselines, training, analysis
scripts/      End-to-end reproduction, training, evaluation, and integrity scripts
docs/         Reproducibility notes, limitations, runtime/cost, hardware, failure recovery
tests/        Engineering and protocol tests for determinism and paper integrity
artifacts/    Generated outputs from local runs
```

## Reproducibility Notes

- Randomness is seeded throughout data generation, evaluation, verifier sampling, and fault injection
- Event ordering and metric reducers are deterministic
- Confidence intervals use clustered bootstrap over stored evaluation records
- Tests cover parser determinism, verifier determinism, event ordering, contamination guards, coverage checks, and trace-to-table reproducibility

For more detail, see:

- `docs/reproducibility.md`
- `docs/failure_recovery.md`
- `docs/hardware_profile.md`
- `docs/runtime_cost.md`
- `docs/limitations_and_design_choices.md`
- `docs/paper_design_choices_and_limitations.md`

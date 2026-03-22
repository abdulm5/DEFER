#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${RUN_DIR:-artifacts/final_run_v1}"
TASKS_PER_DOMAIN="${TASKS_PER_DOMAIN:-300}"
BASELINE_REPEATS="${BASELINE_REPEATS:-3}"
EVAL_REPEATS="${EVAL_REPEATS:-5}"
MAX_SCENARIOS="${MAX_SCENARIOS:-3000}"
SEED="${SEED:-42}"
BASELINE_SPLIT="${BASELINE_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
BOOTSTRAP_MAIN="${BOOTSTRAP_MAIN:-10000}"
BOOTSTRAP_BASELINE="${BOOTSTRAP_BASELINE:-2000}"
MIN_EPISODES_PER_CELL="${MIN_EPISODES_PER_CELL:-100}"
SUCCESS_AUDIT_MAX_TIMING_ALIGNED="${SUCCESS_AUDIT_MAX_TIMING_ALIGNED:-0.70}"
MAX_FALLBACK_RATE="${MAX_FALLBACK_RATE:-0.10}"
CLEAN_OLD="${CLEAN_OLD:-1}"
SKIP_TO_POST_SFT="${SKIP_TO_POST_SFT:-0}"
RUN_API_BASELINE="${RUN_API_BASELINE:-0}"
API_BASE_URL="${API_BASE_URL:-https://api.openai.com/v1/chat/completions}"
API_KEY_ENV="${API_KEY_ENV:-OPENAI_API_KEY}"
API_MODEL="${API_MODEL:-gpt-4o}"
API_POLICY_NAME="${API_POLICY_NAME:-frontier_zero_shot}"
API_REPEATS="${API_REPEATS:-3}"
API_MAX_SCENARIOS="${API_MAX_SCENARIOS:-1000}"
TRAIN_SEED_SWEEP="${TRAIN_SEED_SWEEP:-1}"

if [[ "$SKIP_TO_POST_SFT" == "1" && "$CLEAN_OLD" == "1" ]]; then
  printf "[%s] SKIP_TO_POST_SFT=1: overriding CLEAN_OLD to 0 to preserve existing artifacts.\n" "$(date +"%Y-%m-%d %H:%M:%S")"
  CLEAN_OLD=0
fi

log() {
  printf "[%s] %s\n" "$(date +"%Y-%m-%d %H:%M:%S")" "$*"
}

run() {
  log "RUN: $*"
  "$@"
}

assert_resume_artifacts() {
  if [[ "$SKIP_TO_POST_SFT" != "1" ]]; then
    return
  fi

  local -a required_paths=(
    "$RUN_DIR/training_jobs/full_sft/model"
    "$RUN_DIR/data/training_full/dpo_train.jsonl"
    "$RUN_DIR/data/training_full/dpo_val.jsonl"
    "$RUN_DIR/data/training_full/dpo_perfect_train.jsonl"
    "$RUN_DIR/data/training_full/dpo_perfect_val.jsonl"
    "$RUN_DIR/data/training_success_signal/dpo_train.jsonl"
    "$RUN_DIR/data/training_success_signal/dpo_val.jsonl"
    "$RUN_DIR/data/training_delay_holdout_train/sft_train.jsonl"
    "$RUN_DIR/data/training_delay_holdout_train/sft_val.jsonl"
    "$RUN_DIR/data/training_delay_holdout_train/dpo_train.jsonl"
    "$RUN_DIR/data/training_delay_holdout_train/dpo_val.jsonl"
  )

  local d
  for d in calendar email rest sql webhook file_storage access_control notification; do
    required_paths+=(
      "$RUN_DIR/data/training_domain_holdout_${d}/sft_train.jsonl"
      "$RUN_DIR/data/training_domain_holdout_${d}/sft_val.jsonl"
      "$RUN_DIR/data/training_domain_holdout_${d}/dpo_train.jsonl"
      "$RUN_DIR/data/training_domain_holdout_${d}/dpo_val.jsonl"
    )
  done

  local missing=0
  local path
  for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
      log "Missing required resume artifact: $path"
      missing=1
    fi
  done

  if [[ "$missing" == "1" ]]; then
    log "FAIL: SKIP_TO_POST_SFT=1 but required artifacts are missing."
    exit 12
  fi
}

cleanup_old_artifacts() {
  local run_name
  run_name="$(basename "$RUN_DIR")"

  mkdir -p artifacts

  # Stop stale training/eval processes from prior runs.
  pkill -f "$ROOT_DIR.*scripts.train_sft" >/dev/null 2>&1 || true
  pkill -f "$ROOT_DIR.*scripts.train_dpo" >/dev/null 2>&1 || true
  pkill -f "$ROOT_DIR.*training_queue\\.log" >/dev/null 2>&1 || true
  pkill -f "$ROOT_DIR.*holdout_training_queue\\.log" >/dev/null 2>&1 || true

  for path in artifacts/*; do
    [[ -e "$path" ]] || continue
    if [[ "$(basename "$path")" != "$run_name" ]]; then
      rm -rf "$path"
    fi
  done

  rm -rf "$RUN_DIR"
}

assert_success_pair_audit() {
  python - <<PY
import json
from pathlib import Path
import sys

path = Path("$RUN_DIR/data/success_pair_audit/audit_summary.json")
if not path.exists():
    print(f"Missing audit summary: {path}")
    sys.exit(2)
data = json.loads(path.read_text())
value = float(data.get("timing_aligned_fraction", 1.0))
threshold = float("$SUCCESS_AUDIT_MAX_TIMING_ALIGNED")
print(f"timing_aligned_fraction={value:.6f} threshold={threshold:.6f}")
if value > threshold:
    print("FAIL: success-signal pairs are too entangled with commit-timing signal.")
    sys.exit(3)
PY
}

check_sft_loss_trend() {
  python - <<PY
import re
from pathlib import Path

log_path = Path("$RUN_DIR/training_jobs/logs/full_sft.log")
if not log_path.exists():
    print("WARN: full_sft.log not found; cannot check early loss trend.")
    raise SystemExit(0)

losses = []
for line in log_path.read_text(errors="ignore").splitlines():
    m = re.search(r"(?:\\bloss\\b|train_loss)[^0-9\\-]*([0-9]+\\.?[0-9]*)", line)
    if m:
        losses.append(float(m.group(1)))

if len(losses) < 4:
    print(f"WARN: only {len(losses)} loss points captured; trend check inconclusive.")
    raise SystemExit(0)

head = losses[:4]
tail = losses[-4:]
head_avg = sum(head) / len(head)
tail_avg = sum(tail) / len(tail)
print(f"SFT loss trend check: head_avg={head_avg:.4f} tail_avg={tail_avg:.4f}")
if tail_avg > head_avg * 1.05:
    print("WARN: loss trend did not improve; inspect dataset formatting before trusting results.")
PY
}

assert_dpo_reference_path() {
  local manifest="$1"
  local expected="$2"
  python - <<PY
import json
from pathlib import Path
import sys

manifest = Path("$manifest")
expected = Path("$expected").resolve()
if not manifest.exists():
    print(f"Missing manifest: {manifest}")
    sys.exit(4)
data = json.loads(manifest.read_text())
actual = Path(data.get("model_name", "")).resolve()
print(f"manifest={manifest} model_name={actual}")
if actual != expected:
    print(f"FAIL: expected DPO model_name={expected} but got {actual}")
    sys.exit(5)
PY
}

assert_fallback_rates() {
  local path="$1"
  local max_rate="$2"
  python - <<PY
import pandas as pd
from pathlib import Path
import sys

path = Path("$path")
max_rate = float("$max_rate")
if not path.exists():
    print(f"Missing fallback metrics: {path}")
    sys.exit(6)
df = pd.read_csv(path)
if df.empty:
    print(f"Fallback metrics empty: {path}")
    sys.exit(7)
observed = float(df["fallback_rate"].max())
print(f"max_fallback_rate={observed:.6f} threshold={max_rate:.6f}")
if observed > max_rate:
    print("FAIL: fallback rate exceeded threshold.")
    sys.exit(8)
PY
}

if [[ "$CLEAN_OLD" == "1" ]]; then
  log "Cleaning old artifacts and stale processes."
  cleanup_old_artifacts
fi

assert_resume_artifacts

mkdir -p "$RUN_DIR"/{data,baseline_runs,baseline_metrics,training_jobs/logs,checkpoint_eval,checkpoint_eval_holdouts,api_eval}

if [[ "$SKIP_TO_POST_SFT" == "1" ]]; then
  log "SKIP_TO_POST_SFT=1: skipping data generation/baselines/pair building/SFT and resuming at DPO."
else
  run python -m scripts.generate_seed_tasks \
    --output "$RUN_DIR/data/seed_tasks.jsonl" \
    --tasks-per-domain "$TASKS_PER_DOMAIN" \
    --seed "$SEED"

  run python -m scripts.generate_variants \
    --seed-tasks "$RUN_DIR/data/seed_tasks.jsonl" \
    --output "$RUN_DIR/data/variant_tasks.jsonl" \
    --seed "$SEED"

  run python -m scripts.create_generalization_splits \
    --variants-path "$RUN_DIR/data/variant_tasks.jsonl" \
    --output-dir "$RUN_DIR/data/generalization_splits" \
    --heldout-delay-mechanisms stale_schema_cache,cross_tool_evidence_lag

  run python -m scripts.run_baselines \
    --variants "$RUN_DIR/data/variant_tasks.jsonl" \
    --output-dir "$RUN_DIR/baseline_runs" \
    --split "$BASELINE_SPLIT" \
    --repeats "$BASELINE_REPEATS" \
    --seed "$SEED" \
    --max-scenarios "$MAX_SCENARIOS"

  run python -m scripts.evaluate_metrics \
    --records-path "$RUN_DIR/baseline_runs/reliability_records.jsonl" \
    --output-dir "$RUN_DIR/baseline_metrics" \
    --bootstrap-resamples "$BOOTSTRAP_BASELINE" \
    --seed "$SEED" \
    --min-episodes-per-cell "$MIN_EPISODES_PER_CELL"

  run python -m scripts.build_preference_pairs \
    --traces-path "$RUN_DIR/baseline_runs/episode_traces.jsonl" \
    --output "$RUN_DIR/data/dpo_pairs_triple_mix.jsonl" \
    --include-commit-quality-pairs \
    --target-commit-ratio 0.33 \
    --target-commit-quality-ratio 0.34 \
    --decision-window-turns 5 \
    --min-quality-margin 0.05 \
    --commit-chosen-policies defer_full,perfect_verifier_posttrain,clean_sft_only \
    --commit-quality-chosen-policies defer_full,perfect_verifier_posttrain,clean_sft_only,runtime_verification_only

  run python -m scripts.build_success_preference_pairs \
    --traces-path "$RUN_DIR/baseline_runs/episode_traces.jsonl" \
    --output "$RUN_DIR/data/dpo_pairs_success_signal.jsonl" \
    --target-timing-aligned-ratio 0.6

  run python -m scripts.audit_success_pairs \
    --pairs-path "$RUN_DIR/data/dpo_pairs_success_signal.jsonl" \
    --output-dir "$RUN_DIR/data/success_pair_audit"

  assert_success_pair_audit

  run python -m scripts.build_training_datasets \
    --traces-path "$RUN_DIR/baseline_runs/episode_traces.jsonl" \
    --pairs-path "$RUN_DIR/data/dpo_pairs_triple_mix.jsonl" \
    --output-dir "$RUN_DIR/data/training_full" \
    --seed "$SEED" \
    --val-ratio 0.1

  run python -m scripts.build_training_datasets \
    --traces-path "$RUN_DIR/baseline_runs/episode_traces.jsonl" \
    --pairs-path "$RUN_DIR/data/dpo_pairs_triple_mix.jsonl" \
    --output-dir "$RUN_DIR/data/training_delay_holdout_train" \
    --seed "$SEED" \
    --val-ratio 0.1 \
    --exclude-delay-mechanisms stale_schema_cache,cross_tool_evidence_lag

  for d in calendar email rest sql webhook file_storage access_control notification; do
    run python -m scripts.build_training_datasets \
      --traces-path "$RUN_DIR/baseline_runs/episode_traces.jsonl" \
      --pairs-path "$RUN_DIR/data/dpo_pairs_triple_mix.jsonl" \
      --output-dir "$RUN_DIR/data/training_domain_holdout_${d}" \
      --seed "$SEED" \
      --val-ratio 0.1 \
      --exclude-domains "$d"
  done

  run python -m scripts.build_training_datasets \
    --traces-path "$RUN_DIR/baseline_runs/episode_traces.jsonl" \
    --pairs-path "$RUN_DIR/data/dpo_pairs_success_signal.jsonl" \
    --output-dir "$RUN_DIR/data/training_success_signal" \
    --seed "$SEED" \
    --val-ratio 0.1

  run python -m scripts.train_sft \
    --output-dir "$RUN_DIR/training_jobs/full_sft" \
    --model-name "$BASE_MODEL" \
    --train-path "$RUN_DIR/data/training_full/sft_train.jsonl" \
    --val-path "$RUN_DIR/data/training_full/sft_val.jsonl" \
    --seed "$SEED" \
    --execute 2>&1 | tee "$RUN_DIR/training_jobs/logs/full_sft.log"

  check_sft_loss_trend
fi

run python -m scripts.train_dpo \
  --output-dir "$RUN_DIR/training_jobs/full_dpo_defer" \
  --model-name "$RUN_DIR/training_jobs/full_sft/model" \
  --train-pairs "$RUN_DIR/data/training_full/dpo_train.jsonl" \
  --val-pairs "$RUN_DIR/data/training_full/dpo_val.jsonl" \
  --seed "$SEED" \
  --mode dpo \
  --execute 2>&1 | tee "$RUN_DIR/training_jobs/logs/full_dpo_defer.log"
assert_dpo_reference_path "$RUN_DIR/training_jobs/full_dpo_defer/dpo_manifest.json" "$RUN_DIR/training_jobs/full_sft/model"

run python -m scripts.train_dpo \
  --output-dir "$RUN_DIR/training_jobs/full_dpo_perfect" \
  --model-name "$RUN_DIR/training_jobs/full_sft/model" \
  --train-pairs "$RUN_DIR/data/training_full/dpo_perfect_train.jsonl" \
  --val-pairs "$RUN_DIR/data/training_full/dpo_perfect_val.jsonl" \
  --seed "$SEED" \
  --mode dpo \
  --execute 2>&1 | tee "$RUN_DIR/training_jobs/logs/full_dpo_perfect.log"
assert_dpo_reference_path "$RUN_DIR/training_jobs/full_dpo_perfect/dpo_manifest.json" "$RUN_DIR/training_jobs/full_sft/model"

run python -m scripts.train_dpo \
  --output-dir "$RUN_DIR/training_jobs/full_dpo_success_signal" \
  --model-name "$RUN_DIR/training_jobs/full_sft/model" \
  --train-pairs "$RUN_DIR/data/training_success_signal/dpo_train.jsonl" \
  --val-pairs "$RUN_DIR/data/training_success_signal/dpo_val.jsonl" \
  --seed "$SEED" \
  --mode dpo \
  --execute 2>&1 | tee "$RUN_DIR/training_jobs/logs/full_dpo_success_signal.log"
assert_dpo_reference_path "$RUN_DIR/training_jobs/full_dpo_success_signal/dpo_manifest.json" "$RUN_DIR/training_jobs/full_sft/model"

run python -m scripts.train_sft \
  --output-dir "$RUN_DIR/training_jobs/delay_holdout_sft" \
  --model-name "$BASE_MODEL" \
  --train-path "$RUN_DIR/data/training_delay_holdout_train/sft_train.jsonl" \
  --val-path "$RUN_DIR/data/training_delay_holdout_train/sft_val.jsonl" \
  --seed "$SEED" \
  --execute 2>&1 | tee "$RUN_DIR/training_jobs/logs/delay_holdout_sft.log"

run python -m scripts.train_dpo \
  --output-dir "$RUN_DIR/training_jobs/delay_holdout_dpo_defer" \
  --model-name "$RUN_DIR/training_jobs/delay_holdout_sft/model" \
  --train-pairs "$RUN_DIR/data/training_delay_holdout_train/dpo_train.jsonl" \
  --val-pairs "$RUN_DIR/data/training_delay_holdout_train/dpo_val.jsonl" \
  --seed "$SEED" \
  --mode dpo \
  --execute 2>&1 | tee "$RUN_DIR/training_jobs/logs/delay_holdout_dpo_defer.log"

for d in calendar email rest sql webhook file_storage access_control notification; do
  run python -m scripts.train_sft \
    --output-dir "$RUN_DIR/training_jobs/domain_holdout_${d}_sft" \
    --model-name "$BASE_MODEL" \
    --train-path "$RUN_DIR/data/training_domain_holdout_${d}/sft_train.jsonl" \
    --val-path "$RUN_DIR/data/training_domain_holdout_${d}/sft_val.jsonl" \
    --seed "$SEED" \
    --execute 2>&1 | tee "$RUN_DIR/training_jobs/logs/domain_holdout_${d}_sft.log"

  run python -m scripts.train_dpo \
    --output-dir "$RUN_DIR/training_jobs/domain_holdout_${d}_dpo_defer" \
    --model-name "$RUN_DIR/training_jobs/domain_holdout_${d}_sft/model" \
    --train-pairs "$RUN_DIR/data/training_domain_holdout_${d}/dpo_train.jsonl" \
    --val-pairs "$RUN_DIR/data/training_domain_holdout_${d}/dpo_val.jsonl" \
    --seed "$SEED" \
    --mode dpo \
    --execute 2>&1 | tee "$RUN_DIR/training_jobs/logs/domain_holdout_${d}_dpo_defer.log"
done

# --- Multi-seed training for seed sweep ---
if [[ "$TRAIN_SEED_SWEEP" == "1" ]]; then
  SEEDS_JSON="defer/configs/seeds.json"
  ALL_SEEDS=$(python -c "import json; d=json.load(open('$SEEDS_JSON')); print(' '.join(str(s) for s in d['primary_model_seeds']+d['confirmatory_model_seeds']))")

  for SWEEP_SEED in $ALL_SEEDS; do
    log "Training seed $SWEEP_SEED checkpoints for seed sweep."

    run python -m scripts.train_sft \
      --output-dir "$RUN_DIR/training_jobs/full_sft_seed_${SWEEP_SEED}" \
      --model-name "$BASE_MODEL" \
      --train-path "$RUN_DIR/data/training_full/sft_train.jsonl" \
      --val-path "$RUN_DIR/data/training_full/sft_val.jsonl" \
      --seed "$SWEEP_SEED" \
      --execute 2>&1 | tee "$RUN_DIR/training_jobs/logs/full_sft_seed_${SWEEP_SEED}.log"

    for variant in defer perfect success_signal; do
      case $variant in
        defer)   pairs_train="$RUN_DIR/data/training_full/dpo_train.jsonl"
                 pairs_val="$RUN_DIR/data/training_full/dpo_val.jsonl"
                 suffix="defer" ;;
        perfect) pairs_train="$RUN_DIR/data/training_full/dpo_perfect_train.jsonl"
                 pairs_val="$RUN_DIR/data/training_full/dpo_perfect_val.jsonl"
                 suffix="perfect" ;;
        success_signal) pairs_train="$RUN_DIR/data/training_success_signal/dpo_train.jsonl"
                        pairs_val="$RUN_DIR/data/training_success_signal/dpo_val.jsonl"
                        suffix="success_signal" ;;
      esac

      run python -m scripts.train_dpo \
        --output-dir "$RUN_DIR/training_jobs/full_dpo_${suffix}_seed_${SWEEP_SEED}" \
        --model-name "$RUN_DIR/training_jobs/full_sft_seed_${SWEEP_SEED}/model" \
        --train-pairs "$pairs_train" \
        --val-pairs "$pairs_val" \
        --seed "$SWEEP_SEED" \
        --mode dpo \
        --execute 2>&1 | tee "$RUN_DIR/training_jobs/logs/full_dpo_${suffix}_seed_${SWEEP_SEED}.log"
    done
  done
else
  log "TRAIN_SEED_SWEEP=0: skipping per-seed training."
fi

run python -m scripts.run_checkpoint_eval \
  --variants "$RUN_DIR/data/variant_tasks.jsonl" \
  --output-dir "$RUN_DIR/checkpoint_eval/main" \
  --split "$EVAL_SPLIT" \
  --repeats "$EVAL_REPEATS" \
  --seed "$SEED" \
  --max-scenarios "$MAX_SCENARIOS" \
  --include-baselines runtime_verification_only,react,stress_training_no_contracts \
  --model-policy clean_sft_only="$RUN_DIR/training_jobs/full_sft/model" \
  --model-policy perfect_verifier_posttrain="$RUN_DIR/training_jobs/full_dpo_perfect/model" \
  --model-policy defer_full="$RUN_DIR/training_jobs/full_dpo_defer/model" \
  --model-policy success_signal_posttrain="$RUN_DIR/training_jobs/full_dpo_success_signal/model" \
  --max-fallback-rate "$MAX_FALLBACK_RATE" \
  --training-traces "$RUN_DIR/baseline_runs/episode_traces.jsonl"

run python -m scripts.evaluate_metrics \
  --records-path "$RUN_DIR/checkpoint_eval/main/reliability_records.jsonl" \
  --output-dir "$RUN_DIR/checkpoint_eval/main_metrics" \
  --bootstrap-resamples "$BOOTSTRAP_MAIN" \
  --seed "$SEED" \
  --min-episodes-per-cell "$MIN_EPISODES_PER_CELL" \
  --strict-coverage \
  --fallback-metrics-path "$RUN_DIR/checkpoint_eval/main/fallback_metrics.csv"

if [[ "$TRAIN_SEED_SWEEP" == "1" ]]; then
  run python -m scripts.run_checkpoint_seed_sweep \
    --variants "$RUN_DIR/data/variant_tasks.jsonl" \
    --output-dir "$RUN_DIR/checkpoint_eval/seed_sweep" \
    --seeds-config defer/configs/seeds.json \
    --repeats "$EVAL_REPEATS" \
    --split "$EVAL_SPLIT" \
    --max-scenarios "$MAX_SCENARIOS" \
    --include-baselines runtime_verification_only,react,stress_training_no_contracts \
    --model-policy-template clean_sft_only="$RUN_DIR/training_jobs/full_sft_seed_{seed}/model" \
    --model-policy-template perfect_verifier_posttrain="$RUN_DIR/training_jobs/full_dpo_perfect_seed_{seed}/model" \
    --model-policy-template defer_full="$RUN_DIR/training_jobs/full_dpo_defer_seed_{seed}/model" \
    --model-policy-template success_signal_posttrain="$RUN_DIR/training_jobs/full_dpo_success_signal_seed_{seed}/model"

  run python -m scripts.evaluate_metrics \
    --records-path "$RUN_DIR/checkpoint_eval/seed_sweep/merged_reliability_records.jsonl" \
    --output-dir "$RUN_DIR/checkpoint_eval/seed_sweep_metrics" \
    --bootstrap-resamples "$BOOTSTRAP_MAIN" \
    --seed "$SEED" \
    --min-episodes-per-cell "$MIN_EPISODES_PER_CELL" \
    --strict-coverage \
    --fallback-metrics-path "$RUN_DIR/checkpoint_eval/seed_sweep/fallback_metrics.csv"
else
  log "TRAIN_SEED_SWEEP=0: skipping seed sweep eval (no per-seed checkpoints)."
fi

run python -m scripts.run_checkpoint_eval \
  --variants "$RUN_DIR/data/generalization_splits/delay_holdout/eval_variants.jsonl" \
  --output-dir "$RUN_DIR/checkpoint_eval_holdouts/delay" \
  --split "$EVAL_SPLIT" \
  --repeats "$EVAL_REPEATS" \
  --seed "$SEED" \
  --max-scenarios "$MAX_SCENARIOS" \
  --include-baselines runtime_verification_only \
  --model-policy defer_full="$RUN_DIR/training_jobs/full_dpo_defer/model" \
  --model-policy defer_delay_holdout="$RUN_DIR/training_jobs/delay_holdout_dpo_defer/model" \
  --model-policy success_signal_posttrain="$RUN_DIR/training_jobs/full_dpo_success_signal/model"

run python -m scripts.evaluate_metrics \
  --records-path "$RUN_DIR/checkpoint_eval_holdouts/delay/reliability_records.jsonl" \
  --output-dir "$RUN_DIR/checkpoint_eval_holdouts/delay_metrics" \
  --bootstrap-resamples "$BOOTSTRAP_MAIN" \
  --seed "$SEED" \
  --min-episodes-per-cell "$MIN_EPISODES_PER_CELL" \
  --fallback-metrics-path "$RUN_DIR/checkpoint_eval_holdouts/delay/fallback_metrics.csv"

for d in calendar email rest sql webhook file_storage access_control notification; do
  run python -m scripts.run_checkpoint_eval \
    --variants "$RUN_DIR/data/generalization_splits/domain_holdout/${d}_eval_variants.jsonl" \
    --output-dir "$RUN_DIR/checkpoint_eval_holdouts/domain_${d}" \
    --split "$EVAL_SPLIT" \
    --repeats "$EVAL_REPEATS" \
    --seed "$SEED" \
    --max-scenarios "$MAX_SCENARIOS" \
    --include-baselines runtime_verification_only \
    --model-policy defer_full="$RUN_DIR/training_jobs/full_dpo_defer/model" \
    --model-policy defer_domain_holdout="$RUN_DIR/training_jobs/domain_holdout_${d}_dpo_defer/model" \
    --model-policy success_signal_posttrain="$RUN_DIR/training_jobs/full_dpo_success_signal/model"

  run python -m scripts.evaluate_metrics \
    --records-path "$RUN_DIR/checkpoint_eval_holdouts/domain_${d}/reliability_records.jsonl" \
    --output-dir "$RUN_DIR/checkpoint_eval_holdouts/domain_${d}_metrics" \
    --bootstrap-resamples "$BOOTSTRAP_MAIN" \
    --seed "$SEED" \
    --min-episodes-per-cell "$MIN_EPISODES_PER_CELL" \
    --fallback-metrics-path "$RUN_DIR/checkpoint_eval_holdouts/domain_${d}/fallback_metrics.csv"
done

# Adversarial evaluation
run python -m scripts.run_adversarial_eval \
  --output-dir "$RUN_DIR/checkpoint_eval/adversarial" \
  --repeats "$EVAL_REPEATS" --seed "$SEED" --max-scenarios 200 \
  --include-baselines runtime_verification_only,react \
  --model-policy defer_full="$RUN_DIR/training_jobs/full_dpo_defer/model" \
  --model-policy success_signal_posttrain="$RUN_DIR/training_jobs/full_dpo_success_signal/model"

run python -m scripts.evaluate_metrics \
  --records-path "$RUN_DIR/checkpoint_eval/adversarial/reliability_records.jsonl" \
  --output-dir "$RUN_DIR/checkpoint_eval/adversarial_metrics" \
  --bootstrap-resamples "$BOOTSTRAP_MAIN" \
  --seed "$SEED" \
  --min-episodes-per-cell 10 \
  --fallback-metrics-path "$RUN_DIR/checkpoint_eval/adversarial/fallback_metrics.csv"

# Theory comparison
run python -m scripts.compute_theory_comparison \
  --records-path "$RUN_DIR/checkpoint_eval/main/reliability_records.jsonl" \
  --output-dir "$RUN_DIR/checkpoint_eval/theory" \
  --policy-name defer_full

# Human evaluation preparation
run python -m scripts.prepare_human_eval \
  --traces-path "$RUN_DIR/checkpoint_eval/main/episode_traces.jsonl" \
  --output-dir "$RUN_DIR/human_eval" --n-traces 100 --seed "$SEED"

if [[ "$RUN_API_BASELINE" == "1" ]]; then
  if [[ -z "${!API_KEY_ENV:-}" ]]; then
    log "RUN_API_BASELINE=1 but ${API_KEY_ENV} is not set. Skipping API baseline."
  else
    run python -m scripts.run_api_eval \
      --variants "$RUN_DIR/data/variant_tasks.jsonl" \
      --output-dir "$RUN_DIR/api_eval/openai_main" \
      --split "$EVAL_SPLIT" \
      --repeats "$API_REPEATS" \
      --seed "$SEED" \
      --max-scenarios "$API_MAX_SCENARIOS" \
      --include-baselines runtime_verification_only \
      --api-key-env "$API_KEY_ENV" \
      --base-url "$API_BASE_URL" \
      --api-policy "${API_POLICY_NAME}=${API_MODEL}"

    run python -m scripts.evaluate_metrics \
      --records-path "$RUN_DIR/api_eval/openai_main/reliability_records.jsonl" \
      --output-dir "$RUN_DIR/api_eval/openai_main_metrics" \
      --bootstrap-resamples 5000 \
      --seed "$SEED" \
      --min-episodes-per-cell "$MIN_EPISODES_PER_CELL" \
      --fallback-metrics-path "$RUN_DIR/api_eval/openai_main/fallback_metrics.csv"

    # Prompted deferral baseline
    run python -m scripts.run_api_eval \
      --variants "$RUN_DIR/data/variant_tasks.jsonl" \
      --output-dir "$RUN_DIR/api_eval/prompted_deferral" \
      --split "$EVAL_SPLIT" --repeats "$API_REPEATS" --seed "$SEED" \
      --max-scenarios "$API_MAX_SCENARIOS" \
      --include-baselines runtime_verification_only \
      --api-key-env "$API_KEY_ENV" \
      --base-url "$API_BASE_URL" \
      --api-policy "prompted_deferral=${API_MODEL}" \
      --system-prompt-override prompted_deferral

    run python -m scripts.evaluate_metrics \
      --records-path "$RUN_DIR/api_eval/prompted_deferral/reliability_records.jsonl" \
      --output-dir "$RUN_DIR/api_eval/prompted_deferral_metrics" \
      --bootstrap-resamples 5000 \
      --seed "$SEED" \
      --min-episodes-per-cell "$MIN_EPISODES_PER_CELL" \
      --fallback-metrics-path "$RUN_DIR/api_eval/prompted_deferral/fallback_metrics.csv"
  fi
fi

# Use seed_sweep_metrics if available, otherwise fall back to main_metrics.
if [[ -f "$RUN_DIR/checkpoint_eval/seed_sweep_metrics/claim_gates.json" ]]; then
  CLAIM_GATES_DIR="$RUN_DIR/checkpoint_eval/seed_sweep_metrics"
  FALLBACK_CSV="$RUN_DIR/checkpoint_eval/seed_sweep/fallback_metrics.csv"
else
  CLAIM_GATES_DIR="$RUN_DIR/checkpoint_eval/main_metrics"
  FALLBACK_CSV="$RUN_DIR/checkpoint_eval/main/fallback_metrics.csv"
fi

log "Final claim gates:"
cat "$CLAIM_GATES_DIR/claim_gates.json"

ALLOW_FAILED_CLAIMS="${ALLOW_FAILED_CLAIMS:-0}"
python - <<PY
import json, sys
gates = json.load(open("$CLAIM_GATES_DIR/claim_gates.json"))
failed = []
if not gates.get("gate_1"):
    failed.append("gate_1 (defer_full does not dominate runtime_verification_only)")
if not gates.get("gate_2"):
    failed.append("gate_2 (defer_full DCS does not dominate perfect_verifier)")
if failed:
    for f in failed:
        print(f"FAIL: {f}")
    if "$ALLOW_FAILED_CLAIMS" != "1":
        sys.exit(10)
    print("Continuing because ALLOW_FAILED_CLAIMS=1")
PY

log "Fallback rates:"
python - <<PY
import pandas as pd
path = "$FALLBACK_CSV"
df = pd.read_csv(path)
print(df.sort_values("fallback_rate", ascending=False).to_string(index=False))
PY
assert_fallback_rates "$FALLBACK_CSV" "$MAX_FALLBACK_RATE"

log "Complete. Final outputs are under: $RUN_DIR"

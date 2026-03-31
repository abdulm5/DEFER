from __future__ import annotations

from pathlib import Path


def test_claim_source_selection_keys_off_train_seed_sweep_mode() -> None:
    script = Path("scripts/run_final_data_collection.sh").read_text(encoding="utf-8")
    assert 'if [[ "$TRAIN_SEED_SWEEP" == "1" ]]; then' in script
    assert 'CLAIM_GATES_DIR="$RUN_DIR/checkpoint_eval/seed_sweep_metrics"' in script
    assert 'FALLBACK_CSV="$RUN_DIR/checkpoint_eval/seed_sweep/fallback_metrics.csv"' in script
    assert 'CLAIM_GATES_DIR="$RUN_DIR/checkpoint_eval/main_metrics"' in script
    assert 'FALLBACK_CSV="$RUN_DIR/checkpoint_eval/main/fallback_metrics.csv"' in script
    assert 'if [[ -f "$RUN_DIR/checkpoint_eval/seed_sweep_metrics/claim_gates.json" ]]; then' not in script

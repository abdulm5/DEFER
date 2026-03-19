from __future__ import annotations

EPSILON_GRID = [0.0, 0.1, 0.2, 0.3]
LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3]
K_GRID = [1, 3, 5]

BASELINES = [
    "react",
    "runtime_verification_only",
    "clean_sft_only",
    "stress_training_no_contracts",
    "perfect_verifier_posttrain",
    "defer_full",
]

PRIMARY_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CONFIRMATORY_MODEL = "Qwen/Qwen2.5-7B-Instruct"

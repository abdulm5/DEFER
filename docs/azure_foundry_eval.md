# Azure Foundry Supplementary Evaluation

This guide covers running supplementary SOTA API evaluations with Azure Foundry / Azure OpenAI.

Primary DEFER paper claims are still produced from local checkpoint runs (`checkpoint_eval/main` and optional `seed_sweep`) and are not gated by these supplementary API outputs.

## Endpoint patterns

- Azure OpenAI resource endpoint:
  - `https://YOUR-RESOURCE.openai.azure.com/openai/v1/chat/completions`
- Azure Foundry endpoint:
  - `https://YOUR-RESOURCE.services.ai.azure.com/openai/v1/chat/completions`

If your endpoint requires an API version query parameter, pass:

- `--query-param api-version=2024-10-21` in `run_api_eval`, or
- `query_params: { api-version: "2024-10-21" }` in matrix config.

## Authentication modes

Two auth modes are supported:

- `api_key` (default for Azure examples): sends `<api_key_header>: <token>` (default header `api-key`)
- `bearer`: sends `Authorization: Bearer <token>`

Examples:

```bash
AZURE_OPENAI_API_KEY=... \
python -m scripts.run_api_eval \
  --variants artifacts/final_run_v1/data/variant_tasks.jsonl \
  --output-dir artifacts/final_run_v1/api_eval/azure_zero_shot \
  --split test --repeats 3 --seed 42 --sampling-seed 42 \
  --max-scenarios 1000 \
  --include-baselines runtime_verification_only \
  --api-key-env AZURE_OPENAI_API_KEY \
  --auth-mode api_key \
  --api-key-header api-key \
  --base-url https://YOUR-RESOURCE.openai.azure.com/openai/v1/chat/completions \
  --query-param api-version=2024-10-21 \
  --api-policy frontier_zero_shot=gpt-4o
```

## Parallel matrix runs

Use `run_api_sota_matrix` for supplementary multi-model comparison in parallel.

```bash
python -m scripts.run_api_sota_matrix \
  --config defer/configs/api_sota_matrix.example.yaml \
  --output-root artifacts/final_run_v1/api_sota \
  --workers 2
```

Outputs:

- `matrix_manifest.json`: run configuration, per-model statuses, and paths
- `matrix_summary.csv`: one row per model+variant with status/fallback rates/artifact paths
- per-model subdirs with `zero_shot` and/or `prompted_deferral` eval outputs and metrics

## Integrity checks

Validate primary run only:

```bash
python -m scripts.check_paper_integrity \
  --run-dir artifacts/final_run_v1 \
  --require-seed-sweep
```

Validate primary + supplementary API matrix:

```bash
python -m scripts.check_paper_integrity \
  --run-dir artifacts/final_run_v1 \
  --require-seed-sweep \
  --api-sota-dir artifacts/final_run_v1/api_sota
```

## Common failures and mitigation

- `401 Unauthorized`
  - Token env var not set, wrong auth mode, or wrong header name.
  - Verify `--api-key-env`, `--auth-mode`, and `--api-key-header`.

- `403 Forbidden`
  - Missing model deployment permissions or endpoint-level access policy mismatch.
  - Confirm the model deployment exists and the token has access.

- `429 Too Many Requests`
  - Rate limits exceeded.
  - Reduce matrix `--workers`, lower `--repeats` / `--max-scenarios`, and/or increase retries/backoff:
    - `--max-retries`
    - `--retry-backoff-seconds`
    - `--retry-max-backoff-seconds`

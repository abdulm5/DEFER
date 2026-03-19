# Reproducibility Checklist

## Artifact components

- Source code + pinned dependencies (`requirements-lock.txt`, `pyproject.toml`).
- Environment container (`Dockerfile`).
- Deterministic seeds in data generation and evaluation scripts.
- Raw traces (`episode_traces.jsonl`) and metric records (`reliability_records.jsonl`).
- One-command pipeline (`python -m scripts.reproduce`).

## Determinism controls

- Event scheduling is deterministic (step + insertion order).
- Verifier randomness is seeded per episode.
- Fault injection is seeded per episode.
- Metric reducers are pure functions over stored records.
- Confidence intervals use clustered bootstrap over (`seed`, `scenario_id`) clusters.

## Reproduction command

```bash
python -m scripts.reproduce \
  --output-root artifacts/repro_run \
  --tasks-per-domain 300 \
  --repeats 5 \
  --seed 42
```

## Expected outputs

- `artifacts/repro_run/data/*`
- `artifacts/repro_run/runs/*`
- `artifacts/repro_run/metrics/*`
- `artifacts/repro_run/tables/*`

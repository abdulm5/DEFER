# Expected Runtime and Cost (Mid-Scale)

These are planning estimates for 1-4 GPU setups and can be updated with observed values.

## Stage estimates

- Data generation (1200 seed tasks + variants): 0.5-1.5 CPU hours.
- Baseline simulation/evaluation (full grid, 6 policies): 4-10 CPU hours.
- SFT (7B/8B QLoRA, 2 epochs): 8-20 GPU hours.
- DPO/IPO (1 epoch): 6-14 GPU hours.
- Optional RLVR hardest-10% slice: 6-18 GPU hours.
- Bootstrap CI (10,000 resamples): 0.5-2 CPU hours.

## Approximate spend

- Academic on-prem cluster: no direct cloud cost, ~20-50 GPU-hours total.
- Cloud A100-class reference: roughly $2-$5/hour/GPU => ~$40-$250 depending on runs and repeats.

## Reporting template

For each run, store:
- hardware profile,
- wall-clock duration,
- GPU-hours consumed,
- per-stage cost estimate.

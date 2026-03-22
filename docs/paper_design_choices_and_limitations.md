# Paper Design Choices and Limitations

This document captures the key design choices, explicit limitations, and reporting boundaries for the DEFER paper.

## Integrity Status (Pre-Collection)

As of March 22, 2026, the main integrity-critical issues previously identified in the data collection pipeline are patched:

- Seed-sweep training/eval gating behavior is coherent.
- Checkpoint/API fallback policy confound is removed.
- Strict coverage includes delay-mechanism checks.
- KeyError-side irreversible errors are counted by IER logic.
- Fallback rate NaN validation and contamination guard behavior are hardened.
- Seed sweep now uses eval_seed from the passed seeds config.
- Resume mode has artifact checks to prevent invalid post-SFT resumes.

Operational caveat:

- Training still depends on a runtime environment with train extras installed (`pip install -e ".[train]"`).

## Core Design Choices (And Why)

1. Problem focus: calibrated defer/commit timing under uncertain verification and delayed truth.
- Why: distinguishes DEFER from runtime gating and benchmark-only contributions.

2. Environment substrate: multi-domain, stateful, tool-using simulation.
- Domains include calendar/email/rest/sql plus extended domains for stress realism.
- Why: enough diversity for reliability claims without requiring a large infra stack.

3. Explicit action space.
- `DEFER_WAIT`, `DEFER_REFRESH`, `DEFER_ASK_USER`, `CROSS_CHECK_SECOND_TOOL`, `SAFE_COMMIT_REVERSIBLE`, `FULL_COMMIT_IRREVERSIBLE`.
- Why: enables direct measurement of commitment timing quality.

4. Delayed truth as first-class mechanism.
- Event-loop based reveals and contradiction tracking.
- Why: tests policy behavior when correctness matures after action time.

5. Stress model with two axes.
- Prompt perturbation (`epsilon`) and tool fault intensity (`lambda`).
- Why: aligns with worst-case reliability framing and supports robustness slices.

6. Procedure-aware success.
- Success is reported both raw and gate-aware (corrupt-success aware).
- Why: prevents unsafe trajectories from being over-credited.

7. Training stack separation.
- Clean SFT baseline, DEFER DPO, perfect-verifier DPO, success-signal DPO.
- Why: isolates the effect of defer/commit-targeted supervision.

8. Model-seed protocol.
- 5 primary + 3 confirmatory model seeds.
- Fixed eval sampling seed for sweep comparability.
- Why: separates model variance from scenario-sampling variance.

9. Fallback instrumentation for model policies.
- Parse failures and fallback rate are recorded and gated.
- Why: prevents hidden fallback behavior from inflating reliability claims.

## Limitations and Threats to Validity

1. External validity of synthetic tasks.
- Even with stressors, simulator trajectories may not match enterprise production complexity.
- Mitigation: clearly position as controlled reliability research, not full deployment validation.

2. DCS interpretation under mixed immediate/delayed episodes.
- DCS includes defer and commit F1 components; immediate-only episodes have no defer opportunity.
- Implication: absolute DCS should be interpreted with scenario mix in view.
- Mitigation: report delayed-only and full-mix DCS side by side.

3. Metric denominator conventions.
- IER/EFV are currently normalized by commit actions.
- Implication: reflects per-commit risk, not per-turn prevalence.
- Mitigation: state denominator explicitly in metric definitions.

4. Coverage constraints are protocol-dependent.
- Strict coverage checks episode support by policy/domain/stress/mechanism.
- Implication: claim strength depends on meeting configured minimums.
- Mitigation: publish coverage tables and any failed/relaxed checks.

5. Sequential training wall-clock cost.
- Seeded training is compute-heavy and time-sensitive.
- Mitigation: report exact hardware, runtime, and any truncated runs.

6. API baseline reproducibility.
- Frontier API baselines can drift with provider changes.
- Mitigation: treat API baselines as supplemental; keep primary claims on local checkpoints.

7. Container path is eval-oriented by default.
- Docker default command runs reproduce/eval path, not full training pipeline.
- Mitigation: document native training invocation separately.

## Explicit Non-Claims

DEFER should not claim to have invented:

- asynchronous benchmarking,
- reliability surfaces,
- procedure-aware evaluation,
- verifiable post-training in general.

DEFER should claim:

- calibrated defer/commit learning under uncertain verification and delayed truth,
- delayed-reveal measurement substrate for commitment timing,
- gains on defer/commit reliability metrics versus runtime-gating-only and non-DEFER post-training baselines.

## Reporting Checklist for Final Paper

Include all of the following in the final write-up:

1. Exact data and model seed lists.
2. Exact train/eval split definitions and contamination checks.
3. Coverage tables (cell and delay-mechanism) and minimum-threshold settings.
4. Fallback-rate table per policy.
5. Main and stress-slice metrics with confidence intervals.
6. Procedure-aware and corrupt-success metrics.
7. Ablation table (verifier quality, reveal timing, reversible reward, ask-user reward, cross-check access).
8. Runtime/hardware profile and total GPU-hours.
9. Any deviations from planned pipeline (with reasons).

## Post-Collection Immediate Steps

1. Freeze artifacts and config snapshots used for headline tables.
2. Regenerate summary tables from frozen records only.
3. Run final integrity pass on claim gates, fallback rates, and coverage.
4. Draft limitations paragraph directly from this document and align wording with observed metrics.

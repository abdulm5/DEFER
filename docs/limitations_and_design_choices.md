# DEFER: Limitations and Design Choices

Reference document for the paper's methodology, limitations, and discussion sections.
Organized by what reviewers will ask about.

---

## 1. Simulation Environment

### Design Choices

| Choice | Value | Rationale |
|--------|-------|-----------|
| Max turns per episode | 4 | Keeps episodes tractable; dynamic budget = `min(4, ceil(oracle_turns * 1.5))` |
| Oracle turns | 2 (delayed), 1 (immediate) | Delayed truth requires at least one defer + one commit |
| Policy-invariant RNG | `seed + hash(scenario_id) + repeat * 131` | Ensures paired policy comparisons use identical environment realizations |
| Delayed reveal distribution | Lognormal, category-dependent (A: mu=0.2, B: mu=0.6, C: mu=1.0) | Approximates real-world heavy-tailed event delays |
| Delay step clamp | [1, 6] | Prevents reveals from exceeding practical episode horizons |
| Event drain at episode end | All pending events flushed | Ensures contradiction accounting is policy-invariant |

### Limitations to Acknowledge

- **Fixed turn budget**: 4 turns may be insufficient for deeply nested verification chains. Real tool-use agents often take 10+ turns. Our results characterize short-horizon deferral behavior; longer horizons may exhibit different calibration dynamics.
- **Policy-invariant environment**: All policies experience identical fault/verifier randomness per scenario. This enables clean comparisons but prevents studying adaptive adversarial conditions where the environment responds to policy behavior.
- **Delay distribution is parametric**: Lognormal delays with category-dependent parameters approximate but do not model real asynchronous systems (e.g., exponential backoff, correlated job queues). Category boundaries (epsilon + lambda thresholds at 0.25 and 0.45) are not derived from empirical data.
- **Retroactive contradiction**: Events drained after turn budget can retroactively flip success to failure, which is correct for measuring eventual outcome but may not reflect how practitioners evaluate agents in practice.

---

## 2. Verifier Model

### Design Choices

| Choice | Value | Rationale |
|--------|-------|-----------|
| Decision space | ACCEPT / REJECT / PROVISIONAL | Minimal set capturing certain, failed, and uncertain verification |
| Freshness | Binary (FRESH / STALE), sampled independently | Orthogonal to correctness; models cache staleness |
| Base confidence | 0.9 with penalties: reject -0.5, provisional -0.2, stale -0.25 | Conservative baseline degraded by uncertainty signals |
| Contradiction probability | `base + 0.35*lambda + 0.10*epsilon + 0.08*(delayed)` | Stress-responsive with delayed truth bonus |
| Contradiction sources | Fault-mode-dependent with category fallback | Schema drift -> SCHEMA_CONFLICT, timeout -> DELAYED_JOB, etc. |

### Limitations to Acknowledge

- **Verifier is synthetic**: Uncertainty is injected via parameterized random sampling, not learned from real verification failures. The stale/provisional/contradiction probabilities are manually calibrated. A learned verifier from real tool-use traces would strengthen external validity.
- **Stateless freshness**: Each turn's freshness is sampled independently. Real verification staleness accumulates (e.g., a stale cache remains stale until explicitly refreshed). Our model may underestimate cascading staleness effects.
- **No joint contradiction-verification distribution**: The initial verification decision and the later contradiction are generated independently. In practice, a PROVISIONAL decision may correlate with higher contradiction probability — our model does not capture this.
- **Confidence penalties are not calibrated**: The 0.9 base confidence and fixed penalties (-0.5 for reject, etc.) are not derived from empirical verification accuracy. These values affect the information content of the verifier signal that policies learn from.

---

## 3. Metrics

### Design Choices

| Metric | Formula | Paper Role |
|--------|---------|------------|
| DCS | `0.5 * (F1(defer_prec, defer_rec) + F1(commit_prec, commit_rec))`, per-episode mean | Hero metric |
| IER | `mean(irreversible_errors / total_commit_actions)` per episode | Safety metric |
| EFV | `mean(freshness_violations / total_commit_actions)` per episode | Staleness metric |
| AURS | Unweighted mean of pass@k across (k, epsilon, lambda) grid | Reliability surface volume |
| Gated success | Success AND all procedure gates pass | PAE-style corrupt success filter |

### Limitations to Acknowledge

- **DCS on immediate episodes**: When truth is always resolved (no unresolved events), deferral F1 = 0 by construction. A perfect agent scores DCS = 0.5 on immediate episodes and DCS = 1.0 on delayed episodes. Since all policies are evaluated on the same scenario mix, this bias is constant across comparisons but means raw DCS values are not directly interpretable without conditioning on delay setting.
- **DCS weights over-deferral equally with premature commitment**: The F1-based formulation treats false-positive deferrals (unnecessary waiting) the same as false-negative deferrals (premature commits). In practice, premature irreversible commits may be strictly worse than over-deferral. An asymmetric loss function could be explored in future work.
- **AURS assumes uniform grid importance**: All (k, epsilon, lambda) cells contribute equally. This may not match deployment priorities where high-stress conditions (large epsilon, lambda) matter more.
- **Procedure gates are binary**: Gates pass or fail with no partial credit. `intent_adherence` always passes (no intent verification implemented). `execution_consistency` conflates recoverable retries with unrecoverable budget exhaustion.

---

## 4. Training Pipeline

### Design Choices

| Choice | Value | Rationale |
|--------|-------|-----------|
| Base model | Llama-3.1-8B-Instruct | Instruction-tuned, open-weight, fits 24-48GB GPU with QLoRA |
| SFT method | QLoRA (r=64, alpha=16) on full text (prompt + response) | Memory-efficient; SFT is stage 1, DPO provides DEFER signal |
| DPO method | QLoRA (r=32, alpha=16), beta=0.1, implicit reference (LoRA disabled) | Standard LoRA+DPO; base model = reference when LoRA is off |
| Sequence lengths | SFT: 1024, DPO: 512 (prompt 256 + response 256) | Fits GPU memory constraints |
| Effective batch | SFT: 32 (8 * 4 accum), DPO: 8 (1 * 8 accum) | DPO smaller due to paired examples requiring 2x memory |
| Quantization | 4-bit NF4 with double quantization (bitsandbytes) | Reduces VRAM ~4x; falls back to bf16 if unavailable |
| Epochs | SFT: 2, DPO: 1 | Conservative to avoid overfitting on small preference datasets |

### Limitations to Acknowledge

- **No response masking in SFT**: Loss is computed on all tokens including the user prompt, not just the assistant response. This wastes some gradient on prompt reproduction. Response-only masking could improve efficiency but was not explored.
- **Small effective batch sizes**: DPO effective batch of 8 may introduce gradient noise. Larger batch sizes with gradient accumulation or multi-GPU training could improve optimization stability.
- **Single epoch DPO**: One pass over preference data may underfit, especially on rare pair types. Multi-epoch DPO with early stopping was not explored.
- **Sequence truncation**: DPO max_prompt_length=256 may truncate complex scenario contexts. Multi-turn episodes with several prior actions could exceed this limit.
- **8B models only**: Results may not transfer to larger (70B+) or smaller (1-3B) model scales. Scaling laws for deferral calibration are unknown.

---

## 5. Preference Pair Construction

### Design Choices

| Choice | Value | Rationale |
|--------|-------|-----------|
| Three pair families | Defer-preferred, commit-preferred, commit-quality | Each trains a different aspect of calibrated behavior |
| Quality score | Additive: +2 success, +2 gates, -2.5 corrupt, -0.5/retry, -0.15/turn, ... | Single scalar for ranking trajectories |
| Triple mix ratio | ~33% defer, ~33% commit, ~34% quality | Balanced training signal across deferral dimensions |
| Min quality margin | 0.05 | Filters weak pairs where chosen/rejected are barely distinguishable |
| Decision window | 5 turns | Truncates trace comparison to relevant decision region |

### Limitations to Acknowledge

- **Quality score weights are hand-tuned**: The additive trajectory scoring function (e.g., -0.15 per turn, -0.7 * over_deferral_rate, -0.8 for irreversible commit under uncertainty) uses manually chosen coefficients without principled derivation or sensitivity analysis. Different weight vectors could produce different pair rankings and thus different trained behavior.
- **Pairs are heuristically classified**: The mapping from trajectory properties to pair types (defer_positive, commit_negative, etc.) uses threshold-based rules. These thresholds (e.g., over_deferral_rate <= 0.55 for defer-positive) are not validated against human judgments of deferral quality.
- **Perfect verifier pairs from immediate scenarios**: The "perfect verifier" baseline is trained on pairs where scenario_id ends with `_immediate`, selecting scenarios where verification is trustworthy. This correctly isolates the "no deferred truth" condition but limits the perfect verifier baseline to the immediate-truth distribution, which may differ from the full evaluation distribution.

---

## 6. Experimental Design

### Design Choices

| Choice | Value | Rationale |
|--------|-------|-----------|
| Stress grid | epsilon in {0.0, 0.1, 0.2, 0.3}, lambda in {0.0, 0.1, 0.2, 0.3} | 4x4 grid covers clean-to-moderate stress |
| k values | {1, 3, 5} | Captures single-run through moderate repeated-run reliability |
| Domains | 8 (calendar, email, rest, sql, webhook, file_storage, access_control, notification) | Diverse stateful tool interactions |
| Delay mechanisms | 6 + "none" | Covers eventual consistency, async jobs, concurrent edits, stale caches, delayed auth, cross-tool lag |
| Repeats | 3 (baselines), 5 (eval) | Statistical power for pass@k estimation |
| Seeds | 5 primary + 3 confirmatory | Multi-seed training variance estimation |
| Bootstrap resamples | 10,000 (main), 2,000 (baselines) | Standard for 95% CI estimation |
| Multiple comparisons | Benjamini-Hochberg FDR correction | Controls false discovery rate across pairwise tests |

### Limitations to Acknowledge

- **Moderate stress regime**: Epsilon and lambda max at 0.3. Real production environments can experience catastrophic faults (rate > 0.3) or extreme prompt ambiguity. Our results characterize the moderate-stress regime; extrapolation to extreme conditions is not validated.
- **Deterministic simulation**: The environment is a deterministic simulator, not a live API or production system. While this enables reproducibility and controlled experiments, it cannot capture emergent behaviors of real distributed systems (network partitions, cascading failures, actual eventual consistency).
- **Four domains are core, four are extended**: Calendar, email, REST, and SQL have the deepest tool implementations. Webhook, file_storage, access_control, and notification are extended tools with simpler state management. Domain coverage is broad but not deep.
- **No human-in-the-loop evaluation at scale**: The DEFER_ASK_USER action is simulated as a deferral, not evaluated with actual human responses. Human evaluation is prepared (100 traces) but not integrated into the training loop.

---

## 7. Baselines

### Design Choices

| Baseline | What it represents | Key behavior |
|----------|-------------------|--------------|
| ReAct | No uncertainty awareness | Always commits; no deferral |
| Runtime verification only | ToolGate-style gating | Defers only on REJECT from verifier |
| Clean SFT only | Supervised fine-tuning without deferral signal | Commits based on SFT behavior |
| Perfect verifier post-train | Verifiable training with trusted verifier | Cross-checks once, then commits |
| Success signal post-train | DPO on success/failure pairs (not timing) | Standard outcome-based preference learning |
| DEFER full | Full calibrated deferral training | Our method |

### Limitations to Acknowledge

- **Baselines are handcrafted, not tuned**: The six baseline policies use hardcoded thresholds (e.g., DeferFullPolicy has 11+ manually chosen thresholds). No hyperparameter search was conducted for baselines. This may understate baseline performance.
- **No external model baselines in default run**: The pipeline supports API-based baselines (GPT-4o, prompted deferral) but these are gated behind `RUN_API_BASELINE=1` and not included in the default data collection. Direct comparison with frontier models is optional.
- **Success signal baseline**: The success-signal post-training baseline uses outcome-only preference pairs where timing alignment is controlled (max 60% timing-aligned). This baseline tests whether standard RLHF-style training achieves deferral calibration incidentally. The timing alignment control is verified by audit but the 60% threshold is not derived from a principled analysis.

---

## 8. Statistical Methodology

### Design Choices

- **Clustered bootstrap**: Clusters by (scenario_id, seed) to preserve within-cluster dependence (pass@k structure)
- **Paired clustered bootstrap for diffs**: Only matched clusters across conditions
- **Claim gates**: Non-overlapping 95% CIs required (DEFER > runtime on both AURS and DCS; DEFER DCS > perfect verifier DCS)

### Limitations to Acknowledge

- **Claim gates are conservative**: Requiring non-overlapping CIs is stricter than a standard significance test. This reduces false positives but may fail to detect real improvements when CIs are wide (small sample or high variance).
- **Unmatched clusters silently dropped**: In paired bootstrap, clusters present in only one condition are excluded. If one policy has systematically missing scenarios (e.g., due to crashes), the comparison is on a subset of the evaluation set without warning.
- **No power analysis**: We did not conduct a priori power analysis to determine required sample sizes. Coverage checks enforce minimum episodes per cell but do not guarantee statistical power for the planned comparisons.

---

## Suggested Paper Framing

### What to claim
> We study tool-using agents in settings where verification is uncertain and correctness is only revealed after delayed side effects, and we show that post-training for calibrated deferral and reversible commitment improves worst-case reliability beyond runtime verification alone.

### What NOT to claim
- That the simulator captures all real-world failure modes
- That results transfer to models of different scale
- That the verifier uncertainty model is calibrated to real systems
- That the preference pair scoring function is optimal
- That 4-turn episodes are representative of production tool use

### Key defense points for reviewers
1. **"This is just a simulator"** -> Yes, by design. The contribution is the learning target (calibrated deferral), not the environment. The environment is controlled to isolate the training signal.
2. **"Thresholds are arbitrary"** -> Baselines use the same thresholds across all comparisons. Ablation studies (delayed vs. immediate, reversible reward on/off, cross-check access on/off) isolate the contribution of each design choice.
3. **"Only 8B models"** -> Practical constraint. The method is architecture-agnostic (LoRA + DPO). Scaling studies are future work.
4. **"DCS is a new metric"** -> DCS is the harmonic mean of deferral F1 and commit F1, both standard classification metrics applied to a new decision space. The novelty is in what it measures, not how it computes.

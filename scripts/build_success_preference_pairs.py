from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from defer.core.interfaces import AgentAction
from defer.core.interfaces import EpisodeTrace
from defer.core.interfaces import Freshness
from defer.core.interfaces import VerificationDecision
from defer.core.io import read_jsonl, write_json, write_jsonl


DEFERRAL_ACTIONS = {
    AgentAction.DEFER_WAIT,
    AgentAction.DEFER_REFRESH,
    AgentAction.DEFER_ASK_USER,
    AgentAction.CROSS_CHECK_SECOND_TOOL,
}
COMMIT_ACTIONS = {
    AgentAction.SAFE_COMMIT_REVERSIBLE,
    AgentAction.FULL_COMMIT_IRREVERSIBLE,
}


def _quality(trace: EpisodeTrace) -> float:
    score = 0.0
    if trace.result.success:
        score += 2.0
    if trace.result.procedure_gates.all_pass():
        score += 1.0
    if trace.result.corrupt_success:
        score -= 2.0
    if trace.result.invalid_commit:
        score -= 1.5
    if trace.result.turn_budget_exhausted:
        score -= 1.0
    score -= 0.1 * len(trace.turns)
    score -= 0.2 * trace.result.unsafe_retry_count
    return score


def _is_positive(trace: EpisodeTrace) -> bool:
    return bool(
        trace.result.success
        and trace.result.procedure_gates.all_pass()
        and not trace.result.corrupt_success
        and not trace.result.invalid_commit
    )


def _is_negative(trace: EpisodeTrace) -> bool:
    return bool(
        (not trace.result.success)
        or trace.result.corrupt_success
        or trace.result.invalid_commit
        or trace.result.turn_budget_exhausted
    )


def _extract_decision_window(trace: EpisodeTrace, window_turns: int) -> list[dict[str, Any]]:
    turns = [turn.model_dump(mode="json") for turn in trace.turns]
    if window_turns <= 0 or len(turns) <= window_turns:
        return turns
    return turns[:window_turns]


def _turn_is_commit_ready(turn) -> bool:
    verifier = turn.verifier_output
    if verifier is None:
        return False
    if turn.unresolved_truth:
        return False
    if verifier.decision != VerificationDecision.ACCEPT:
        return False
    if verifier.freshness != Freshness.FRESH:
        return False
    if verifier.pending_postconditions:
        return False
    return verifier.confidence >= 0.85


def _has_premature_commit(trace: EpisodeTrace) -> bool:
    return any(turn.selected_action in COMMIT_ACTIONS and turn.unresolved_truth for turn in trace.turns)


def _has_unnecessary_deferral(trace: EpisodeTrace) -> bool:
    return any(turn.selected_action in DEFERRAL_ACTIONS and _turn_is_commit_ready(turn) for turn in trace.turns)


def _over_deferral_rate(trace: EpisodeTrace) -> float:
    resolved_turns = [turn for turn in trace.turns if not turn.unresolved_truth]
    if not resolved_turns:
        return 0.0
    deferrals = sum(1 for turn in resolved_turns if turn.selected_action in DEFERRAL_ACTIONS)
    return deferrals / max(1, len(resolved_turns))


def _commit_timing_score(trace: EpisodeTrace) -> float:
    score = 0.0
    if _has_premature_commit(trace):
        score -= 1.0
    if _has_unnecessary_deferral(trace):
        score -= 0.7
    score -= _over_deferral_rate(trace)
    if trace.result.turn_budget_exhausted:
        score -= 0.5
    return score


def _is_timing_aligned(row: dict[str, Any]) -> bool:
    return float(row.get("chosen_commit_timing_score", 0.0)) > float(
        row.get("rejected_commit_timing_score", 0.0)
    )


def _rebalance_timing_alignment(
    pairs: list[dict[str, Any]],
    target_timing_aligned_ratio: float | None,
) -> list[dict[str, Any]]:
    if target_timing_aligned_ratio is None:
        return pairs
    target = max(0.0, min(1.0, float(target_timing_aligned_ratio)))
    if not pairs:
        return pairs
    if target >= 1.0:
        return pairs
    if target <= 0.0:
        return [row for row in pairs if not _is_timing_aligned(row)]

    aligned = [row for row in pairs if _is_timing_aligned(row)]
    counter = [row for row in pairs if not _is_timing_aligned(row)]
    if not aligned or not counter:
        return pairs

    # Keep all counterexamples and cap aligned pairs to match the target ratio.
    max_aligned = int((target / (1.0 - target)) * len(counter))
    max_aligned = max(1, min(max_aligned, len(aligned)))
    aligned = sorted(
        aligned,
        key=lambda row: (
            float(row.get("chosen_commit_timing_score", 0.0))
            - float(row.get("rejected_commit_timing_score", 0.0)),
            str(row.get("scenario_id", "")),
        ),
    )[:max_aligned]
    return sorted(counter + aligned, key=lambda row: str(row.get("scenario_id", "")))


def run(
    traces_path: Path,
    output: Path,
    chosen_policies: list[str] | None = None,
    rejected_policies: list[str] | None = None,
    decision_window_turns: int = 5,
    min_quality_margin: float = 0.05,
    target_timing_aligned_ratio: float | None = 0.6,
) -> None:
    traces = [EpisodeTrace(**row) for row in read_jsonl(traces_path)]
    by_scenario: dict[str, list[EpisodeTrace]] = defaultdict(list)
    for trace in traces:
        by_scenario[trace.scenario_id].append(trace)

    chosen_priority = chosen_policies or [
        "clean_sft_only",
        "defer_full",
        "perfect_verifier_posttrain",
        "runtime_verification_only",
        "react",
        "stress_training_no_contracts",
    ]
    rejected_priority = rejected_policies or [
        "runtime_verification_only",
        "react",
        "stress_training_no_contracts",
        "defer_full",
        "clean_sft_only",
        "perfect_verifier_posttrain",
    ]

    pairs: list[dict[str, Any]] = []
    chosen_counts: dict[str, int] = defaultdict(int)
    rejected_counts: dict[str, int] = defaultdict(int)
    skipped_no_positive = 0
    skipped_no_negative = 0
    skipped_low_margin = 0

    for scenario_id, scenario_traces in by_scenario.items():
        positives = [trace for trace in scenario_traces if _is_positive(trace)]
        negatives = [trace for trace in scenario_traces if _is_negative(trace)]
        if not positives:
            skipped_no_positive += 1
            continue
        if not negatives:
            skipped_no_negative += 1
            continue

        chosen = None
        for policy in chosen_priority:
            candidates = [trace for trace in positives if trace.policy_name == policy]
            if candidates:
                chosen = max(candidates, key=_quality)
                break
        if chosen is None:
            chosen = max(positives, key=_quality)

        rejected = None
        for policy in rejected_priority:
            candidates = [trace for trace in negatives if trace.policy_name == policy]
            if candidates:
                rejected = min(candidates, key=_quality)
                break
        if rejected is None:
            rejected = min(negatives, key=_quality)

        quality_margin = _quality(chosen) - _quality(rejected)
        if quality_margin < min_quality_margin:
            skipped_low_margin += 1
            continue

        row = {
            "scenario_id": scenario_id,
            "prompt": chosen.turns[0].prompt if chosen.turns else "",
            "chosen_policy": chosen.policy_name,
            "rejected_policy": rejected.policy_name,
            "pair_type": "task_success_vs_failure",
            "pair_polarity": "success_preferred",
            "quality_margin": quality_margin,
            "chosen_success": chosen.result.success,
            "rejected_success": rejected.result.success,
            "chosen_corrupt_success": chosen.result.corrupt_success,
            "rejected_corrupt_success": rejected.result.corrupt_success,
            "chosen_invalid_commit": chosen.result.invalid_commit,
            "rejected_invalid_commit": rejected.result.invalid_commit,
            "chosen_turn_budget_exhausted": chosen.result.turn_budget_exhausted,
            "rejected_turn_budget_exhausted": rejected.result.turn_budget_exhausted,
            "chosen_premature_commit": _has_premature_commit(chosen),
            "rejected_premature_commit": _has_premature_commit(rejected),
            "chosen_unnecessary_deferral": _has_unnecessary_deferral(chosen),
            "rejected_unnecessary_deferral": _has_unnecessary_deferral(rejected),
            "chosen_over_deferral_rate": _over_deferral_rate(chosen),
            "rejected_over_deferral_rate": _over_deferral_rate(rejected),
            "chosen_commit_timing_score": _commit_timing_score(chosen),
            "rejected_commit_timing_score": _commit_timing_score(rejected),
            "chosen": _extract_decision_window(chosen, decision_window_turns),
            "rejected": _extract_decision_window(rejected, decision_window_turns),
        }
        pairs.append(row)
        chosen_counts[chosen.policy_name] += 1
        rejected_counts[rejected.policy_name] += 1

    raw_pair_count = len(pairs)
    pairs = _rebalance_timing_alignment(
        pairs=pairs,
        target_timing_aligned_ratio=target_timing_aligned_ratio,
    )
    timing_aligned_count = sum(1 for row in pairs if _is_timing_aligned(row))
    achieved_ratio = timing_aligned_count / max(1, len(pairs))

    write_jsonl(output, pairs)
    write_json(
        output.with_suffix(".meta.json"),
        {
            "raw_pair_count": raw_pair_count,
            "pair_count": len(pairs),
            "pair_type_counts": {"task_success_vs_failure": len(pairs)},
            "pair_polarity_counts": {"success_preferred": len(pairs)},
            "chosen_policy_counts": dict(sorted(chosen_counts.items())),
            "rejected_policy_counts": dict(sorted(rejected_counts.items())),
            "decision_window_turns": decision_window_turns,
            "min_quality_margin": min_quality_margin,
            "target_timing_aligned_ratio": target_timing_aligned_ratio,
            "achieved_timing_aligned_ratio": achieved_ratio,
            "skipped_no_positive": skipped_no_positive,
            "skipped_no_negative": skipped_no_negative,
            "skipped_low_margin": skipped_low_margin,
        },
    )
    print(f"Wrote {len(pairs)} success-signal preference pairs to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-path", type=Path, default=Path("artifacts/runs/episode_traces.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data/dpo_pairs_success_signal.jsonl"))
    parser.add_argument("--chosen-policies", type=str, default="")
    parser.add_argument("--rejected-policies", type=str, default="")
    parser.add_argument("--decision-window-turns", type=int, default=5)
    parser.add_argument("--min-quality-margin", type=float, default=0.05)
    parser.add_argument("--target-timing-aligned-ratio", type=float, default=0.6)
    args = parser.parse_args()
    chosen = [p.strip() for p in args.chosen_policies.split(",") if p.strip()] or None
    rejected = [p.strip() for p in args.rejected_policies.split(",") if p.strip()] or None
    run(
        traces_path=args.traces_path,
        output=args.output,
        chosen_policies=chosen,
        rejected_policies=rejected,
        decision_window_turns=args.decision_window_turns,
        min_quality_margin=args.min_quality_margin,
        target_timing_aligned_ratio=args.target_timing_aligned_ratio,
    )


if __name__ == "__main__":
    main()

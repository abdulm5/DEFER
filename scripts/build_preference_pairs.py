from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from defer.core.interfaces import AgentAction
from defer.core.interfaces import EpisodeTrace
from defer.core.interfaces import Freshness
from defer.core.interfaces import VerificationDecision
from defer.core.io import read_jsonl, write_json, write_jsonl


DEFAULT_CHOSEN_POLICIES = ["defer_full"]
DEFAULT_REJECTED_POLICIES = [
    "runtime_verification_only",
    "react",
    "clean_sft_only",
    "stress_training_no_contracts",
    "perfect_verifier_posttrain",
]
DEFAULT_COMMIT_CHOSEN_POLICIES = [
    "defer_full",
    "perfect_verifier_posttrain",
    "clean_sft_only",
]
DEFAULT_COMMIT_QUALITY_CHOSEN_POLICIES = [
    "defer_full",
    "perfect_verifier_posttrain",
    "clean_sft_only",
    "runtime_verification_only",
]

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
UNCERTAIN_CONFIDENCE_MAX = 0.70
COMMIT_READY_CONFIDENCE_MIN = 0.85
COMMIT_POLARITY = "commit_preferred"
DEFER_POLARITY = "defer_preferred"
COMMIT_QUALITY_POLARITY = "commit_quality_preferred"


def _trajectory_quality(trace: EpisodeTrace) -> float:
    """
    Higher is better.

    Balances safety/integrity with deferral calibration and turn efficiency.
    """
    score = 0.0
    if trace.result.success:
        score += 2.0
    if trace.result.procedure_gates.all_pass():
        score += 2.0
    if trace.result.corrupt_success:
        score -= 2.5
    if trace.result.invalid_commit:
        score -= 2.0
    score -= 0.5 * trace.result.unsafe_retry_count
    score -= 0.15 * len(trace.turns)
    if trace.result.turn_budget_exhausted:
        score -= 1.5
    score -= 0.7 * _over_deferral_rate(trace)
    score -= 0.25 * _decision_latency(trace)

    for turn in trace.turns:
        if _turn_is_uncertain(turn) and turn.selected_action in DEFERRAL_ACTIONS:
            score += 0.2
        if _turn_is_commit_ready(turn) and turn.selected_action in DEFERRAL_ACTIONS:
            score -= 0.35
        if turn.irreversible_commit and _turn_is_uncertain(turn):
            score -= 0.5
        if (
            _turn_is_uncertain(turn)
            and turn.selected_action == AgentAction.FULL_COMMIT_IRREVERSIBLE
            and turn.irreversible_commit
        ):
            score -= 0.8
        if turn.used_stale_evidence:
            score -= 0.2
    return score


def _base_positive(trace: EpisodeTrace) -> bool:
    return bool(
        trace.result.success
        and trace.result.procedure_gates.all_pass()
        and not trace.result.corrupt_success
        and not trace.result.invalid_commit
    )


def _is_defer_positive(trace: EpisodeTrace) -> bool:
    return bool(
        _base_positive(trace)
        and _has_calibrated_deferral(trace)
        and _over_deferral_rate(trace) <= 0.55
    )


def _is_defer_negative(trace: EpisodeTrace) -> bool:
    if not trace.result.success:
        return True
    if trace.result.corrupt_success or trace.result.invalid_commit:
        return True
    if _has_premature_commit(trace):
        return True
    if _has_stale_commit(trace):
        return True
    return False


def _is_commit_positive(trace: EpisodeTrace) -> bool:
    return bool(
        _base_positive(trace)
        and _has_commit_ready_commit(trace)
        and _over_deferral_rate(trace) <= 0.15
        and _decision_latency(trace) <= 1
        and not trace.result.turn_budget_exhausted
    )


def _is_commit_negative(trace: EpisodeTrace) -> bool:
    if _over_deferral_rate(trace) >= 0.25:
        return True
    if trace.result.turn_budget_exhausted:
        return True
    if _has_unnecessary_deferral(trace):
        return True
    if _decision_latency(trace) >= 2:
        return True
    return False


def _is_commit_quality_positive(trace: EpisodeTrace) -> bool:
    return bool(
        _base_positive(trace)
        and _has_commit_ready_commit(trace)
        and not _has_premature_commit(trace)
        and not _has_stale_commit(trace)
        and not trace.result.turn_budget_exhausted
    )


def _is_commit_quality_negative(trace: EpisodeTrace) -> bool:
    if trace.result.corrupt_success or not trace.result.procedure_gates.all_pass():
        return True
    if _has_premature_commit(trace):
        return True
    if _has_stale_commit(trace):
        return True
    if trace.result.turn_budget_exhausted:
        return True
    return False


def _over_deferral_rate(trace: EpisodeTrace) -> float:
    deferrals_on_resolved = sum(
        1 for turn in trace.turns if (not turn.unresolved_truth) and turn.selected_action in DEFERRAL_ACTIONS
    )
    resolved_turns = max(1, sum(1 for turn in trace.turns if not turn.unresolved_truth))
    return deferrals_on_resolved / resolved_turns


def _turn_is_uncertain(turn) -> bool:
    verifier = turn.verifier_output
    if turn.unresolved_truth:
        return True
    if verifier is None:
        return True
    if verifier.decision != VerificationDecision.ACCEPT:
        return True
    if verifier.freshness == Freshness.STALE:
        return True
    if verifier.pending_postconditions:
        return True
    return verifier.confidence < UNCERTAIN_CONFIDENCE_MAX


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
    return verifier.confidence >= COMMIT_READY_CONFIDENCE_MIN


def _has_calibrated_deferral(trace: EpisodeTrace) -> bool:
    return any(
        turn.selected_action in DEFERRAL_ACTIONS and _turn_is_uncertain(turn)
        for turn in trace.turns
    )


def _has_commit_ready_commit(trace: EpisodeTrace) -> bool:
    return any(
        turn.selected_action in COMMIT_ACTIONS and _turn_is_commit_ready(turn)
        for turn in trace.turns
    )


def _has_unnecessary_deferral(trace: EpisodeTrace) -> bool:
    return any(
        turn.selected_action in DEFERRAL_ACTIONS and _turn_is_commit_ready(turn)
        for turn in trace.turns
    )


def _has_premature_commit(trace: EpisodeTrace) -> bool:
    return any(
        turn.selected_action in COMMIT_ACTIONS and _turn_is_uncertain(turn)
        for turn in trace.turns
    )


def _has_stale_commit(trace: EpisodeTrace) -> bool:
    return any(
        turn.selected_action in COMMIT_ACTIONS and turn.used_stale_evidence
        for turn in trace.turns
    )


def _decision_latency(trace: EpisodeTrace) -> int:
    first_ready_idx: int | None = None
    first_commit_idx: int | None = None
    for idx, turn in enumerate(trace.turns):
        if first_ready_idx is None and _turn_is_commit_ready(turn):
            first_ready_idx = idx
        if first_ready_idx is not None and turn.selected_action in COMMIT_ACTIONS:
            first_commit_idx = idx
            break
    if first_ready_idx is None:
        return 0
    if first_commit_idx is None:
        return len(trace.turns) - first_ready_idx
    return max(0, first_commit_idx - first_ready_idx)


def _decision_turn_index(trace: EpisodeTrace) -> int:
    for idx, turn in enumerate(trace.turns):
        if turn.selected_action in DEFERRAL_ACTIONS | COMMIT_ACTIONS and (
            _turn_is_uncertain(turn) or _turn_is_commit_ready(turn)
        ):
            return idx
    for idx, turn in enumerate(trace.turns):
        if turn.selected_action in DEFERRAL_ACTIONS | COMMIT_ACTIONS:
            return idx
    return 0


def _extract_decision_window(trace: EpisodeTrace, window_turns: int) -> tuple[list[dict], int]:
    turns = [turn.model_dump(mode="json") for turn in trace.turns]
    if window_turns <= 0 or len(turns) <= window_turns:
        return turns, _decision_turn_index(trace)
    center = _decision_turn_index(trace)
    half = max(0, window_turns // 2)
    start = max(0, center - half)
    end = start + window_turns
    if end > len(turns):
        end = len(turns)
        start = max(0, end - window_turns)
    return turns[start:end], center


def _pair_type(chosen: EpisodeTrace, rejected: EpisodeTrace, polarity: str) -> str:
    if polarity == COMMIT_QUALITY_POLARITY:
        if rejected.result.corrupt_success:
            return "careful_commit_vs_corrupt_commit"
        if rejected.result.turn_budget_exhausted:
            return "careful_commit_vs_rushed_budget_exhaustion"
        if _has_stale_commit(rejected):
            return "careful_commit_vs_stale_commit"
        if _has_premature_commit(rejected):
            return "careful_commit_vs_premature_commit"
        return "careful_commit_vs_low_integrity_commit"

    if polarity == COMMIT_POLARITY:
        if rejected.result.turn_budget_exhausted:
            return "turn_budget_exhaustion_vs_timely_commit"
        if _has_unnecessary_deferral(rejected):
            return "unnecessary_deferral_vs_timely_commit"
        return "over_deferral_vs_timely_commit"

    rejected_has_premature_irreversible = any(
        turn.unresolved_truth
        and turn.selected_action == AgentAction.FULL_COMMIT_IRREVERSIBLE
        and turn.irreversible_commit
        for turn in rejected.turns
    )
    chosen_has_cross_check = any(
        turn.selected_action in {AgentAction.DEFER_REFRESH, AgentAction.CROSS_CHECK_SECOND_TOOL}
        for turn in chosen.turns
    )
    rejected_over_defer = _over_deferral_rate(rejected) >= 0.75
    chosen_over_defer = _over_deferral_rate(chosen) >= 0.75
    if rejected_has_premature_irreversible and chosen_has_cross_check:
        return "premature_irreversible_commit_vs_calibrated_deferral"
    if rejected_over_defer and not chosen_over_defer:
        return "over_deferral_vs_timely_commit"
    if any(turn.used_stale_evidence for turn in rejected.turns):
        return "stale_evidence_commit_vs_refresh"
    return "general_commit_timing"


def _choose_trace(
    traces: list[EpisodeTrace],
    policy_priority: list[str],
    predicate,
    choose_max: bool,
    allow_fallback: bool,
) -> EpisodeTrace | None:
    for policy_name in policy_priority:
        candidates = [t for t in traces if t.policy_name == policy_name and predicate(t)]
        if candidates:
            return (
                max(candidates, key=_trajectory_quality)
                if choose_max
                else min(candidates, key=_trajectory_quality)
            )

    if not allow_fallback:
        return None
    fallback = [t for t in traces if predicate(t)]
    if not fallback:
        return None
    return max(fallback, key=_trajectory_quality) if choose_max else min(fallback, key=_trajectory_quality)


def _build_pair_row(
    scenario_id: str,
    chosen: EpisodeTrace,
    rejected: EpisodeTrace,
    pair_polarity: str,
    window_turns: int,
) -> dict:
    chosen_turns, chosen_decision_turn = _extract_decision_window(chosen, window_turns=window_turns)
    rejected_turns, rejected_decision_turn = _extract_decision_window(rejected, window_turns=window_turns)
    chosen_quality = _trajectory_quality(chosen)
    rejected_quality = _trajectory_quality(rejected)
    return {
        "scenario_id": scenario_id,
        "prompt": chosen.turns[0].prompt if chosen.turns else "",
        "chosen_policy": chosen.policy_name,
        "rejected_policy": rejected.policy_name,
        "pair_type": _pair_type(chosen, rejected, polarity=pair_polarity),
        "pair_polarity": pair_polarity,
        "chosen_quality": chosen_quality,
        "rejected_quality": rejected_quality,
        "quality_margin": chosen_quality - rejected_quality,
        "chosen_over_deferral_rate": _over_deferral_rate(chosen),
        "rejected_over_deferral_rate": _over_deferral_rate(rejected),
        "chosen_total_turns": len(chosen.turns),
        "rejected_total_turns": len(rejected.turns),
        "chosen_decision_turn": chosen_decision_turn,
        "rejected_decision_turn": rejected_decision_turn,
        "decision_window_turns": window_turns,
        "chosen": chosen_turns,
        "rejected": rejected_turns,
    }


def _balance_by_polarity(
    pairs: list[dict],
    target_commit_ratio: float,
    target_commit_quality_ratio: float = 0.0,
) -> list[dict]:
    commit_ratio = min(max(target_commit_ratio, 0.0), 1.0)
    quality_ratio = min(max(target_commit_quality_ratio, 0.0), 1.0)
    if commit_ratio + quality_ratio > 1.0:
        raise ValueError("target_commit_ratio + target_commit_quality_ratio must be <= 1.0")
    defer_ratio = 1.0 - commit_ratio - quality_ratio

    by_polarity: dict[str, list[dict]] = {}
    for polarity in {DEFER_POLARITY, COMMIT_POLARITY, COMMIT_QUALITY_POLARITY}:
        by_polarity[polarity] = sorted(
            [pair for pair in pairs if pair.get("pair_polarity") == polarity],
            key=lambda pair: (pair.get("quality_margin", 0.0), pair.get("scenario_id", "")),
            reverse=True,
        )

    active_targets = {
        DEFER_POLARITY: defer_ratio,
        COMMIT_POLARITY: commit_ratio,
        COMMIT_QUALITY_POLARITY: quality_ratio,
    }
    active_targets = {
        polarity: ratio for polarity, ratio in active_targets.items() if ratio > 0.0 and by_polarity[polarity]
    }
    if not active_targets:
        return sorted(pairs, key=lambda pair: pair.get("scenario_id", ""))

    totals: list[int] = []
    for polarity, ratio in active_targets.items():
        if ratio <= 0.0:
            continue
        totals.append(int(len(by_polarity[polarity]) / ratio))
    total = min(totals) if totals else 0
    if total <= 0:
        return sorted(pairs, key=lambda pair: pair.get("scenario_id", ""))

    keep: dict[str, int] = {}
    for polarity, ratio in active_targets.items():
        keep[polarity] = min(len(by_polarity[polarity]), max(1, round(total * ratio)))

    while sum(keep.values()) > total:
        largest = max(keep, key=lambda p: keep[p])
        if keep[largest] > 1:
            keep[largest] -= 1
        else:
            break

    while sum(keep.values()) < total:
        can_add = [
            polarity
            for polarity in active_targets
            if keep[polarity] < len(by_polarity[polarity])
        ]
        if not can_add:
            break
        can_add.sort(
            key=lambda p: keep[p] / max(active_targets[p], 1e-9)
        )
        keep[can_add[0]] += 1

    selected: list[dict] = []
    for polarity, count in keep.items():
        selected.extend(by_polarity[polarity][:count])
    return sorted(selected, key=lambda pair: pair.get("scenario_id", ""))


def run(
    traces_path: Path,
    output: Path,
    chosen_policies: list[str] | None = None,
    commit_chosen_policies: list[str] | None = None,
    commit_quality_chosen_policies: list[str] | None = None,
    rejected_policies: list[str] | None = None,
    allow_same_policy: bool = False,
    allow_chosen_fallback: bool = False,
    allow_commit_chosen_fallback: bool = False,
    allow_commit_quality_chosen_fallback: bool = False,
    include_commit_quality_pairs: bool = False,
    target_commit_ratio: float = 0.5,
    target_commit_quality_ratio: float = 0.0,
    decision_window_turns: int = 5,
    min_quality_margin: float = 0.05,
) -> None:
    traces = [EpisodeTrace(**row) for row in read_jsonl(traces_path)]
    by_scenario: dict[str, list[EpisodeTrace]] = defaultdict(list)
    for trace in traces:
        by_scenario[trace.scenario_id].append(trace)

    chosen_priority = chosen_policies or DEFAULT_CHOSEN_POLICIES
    commit_chosen_priority = commit_chosen_policies or DEFAULT_COMMIT_CHOSEN_POLICIES
    commit_quality_chosen_priority = (
        commit_quality_chosen_policies or DEFAULT_COMMIT_QUALITY_CHOSEN_POLICIES
    )
    rejected_priority = rejected_policies or DEFAULT_REJECTED_POLICIES

    pairs: list[dict] = []
    chosen_counts: dict[str, int] = defaultdict(int)
    rejected_counts: dict[str, int] = defaultdict(int)
    pair_type_counts: dict[str, int] = defaultdict(int)
    pair_polarity_counts: dict[str, int] = defaultdict(int)
    skipped_no_positive = 0
    skipped_no_negative = 0
    skipped_same_policy = 0
    skipped_low_margin = 0

    for scenario_id, scenario_traces in by_scenario.items():
        defer_chosen = _choose_trace(
            traces=scenario_traces,
            policy_priority=chosen_priority,
            predicate=_is_defer_positive,
            choose_max=True,
            allow_fallback=allow_chosen_fallback,
        )
        defer_rejected = _choose_trace(
            traces=scenario_traces,
            policy_priority=rejected_priority,
            predicate=_is_defer_negative,
            choose_max=False,
            allow_fallback=True,
        )
        if defer_chosen is not None and defer_rejected is not None:
            if (not allow_same_policy) and defer_chosen.policy_name == defer_rejected.policy_name:
                skipped_same_policy += 1
            else:
                row = _build_pair_row(
                    scenario_id=scenario_id,
                    chosen=defer_chosen,
                    rejected=defer_rejected,
                    pair_polarity=DEFER_POLARITY,
                    window_turns=decision_window_turns,
                )
                if row["quality_margin"] >= min_quality_margin:
                    pairs.append(row)
                    chosen_counts[defer_chosen.policy_name] += 1
                    rejected_counts[defer_rejected.policy_name] += 1
                    pair_type_counts[row["pair_type"]] += 1
                    pair_polarity_counts[DEFER_POLARITY] += 1
                else:
                    skipped_low_margin += 1
        else:
            if defer_chosen is None:
                skipped_no_positive += 1
            if defer_rejected is None:
                skipped_no_negative += 1

        commit_chosen = _choose_trace(
            traces=scenario_traces,
            policy_priority=commit_chosen_priority,
            predicate=_is_commit_positive,
            choose_max=True,
            allow_fallback=allow_commit_chosen_fallback,
        )
        commit_rejected = _choose_trace(
            traces=scenario_traces,
            policy_priority=rejected_priority,
            predicate=_is_commit_negative,
            choose_max=False,
            allow_fallback=True,
        )
        if commit_chosen is not None and commit_rejected is not None:
            if (not allow_same_policy) and commit_chosen.policy_name == commit_rejected.policy_name:
                skipped_same_policy += 1
            else:
                row = _build_pair_row(
                    scenario_id=scenario_id,
                    chosen=commit_chosen,
                    rejected=commit_rejected,
                    pair_polarity=COMMIT_POLARITY,
                    window_turns=decision_window_turns,
                )
                if row["quality_margin"] >= min_quality_margin:
                    pairs.append(row)
                    chosen_counts[commit_chosen.policy_name] += 1
                    rejected_counts[commit_rejected.policy_name] += 1
                    pair_type_counts[row["pair_type"]] += 1
                    pair_polarity_counts[COMMIT_POLARITY] += 1
                else:
                    skipped_low_margin += 1

        if include_commit_quality_pairs:
            quality_chosen = _choose_trace(
                traces=scenario_traces,
                policy_priority=commit_quality_chosen_priority,
                predicate=_is_commit_quality_positive,
                choose_max=True,
                allow_fallback=allow_commit_quality_chosen_fallback,
            )
            quality_rejected = _choose_trace(
                traces=scenario_traces,
                policy_priority=rejected_priority,
                predicate=_is_commit_quality_negative,
                choose_max=False,
                allow_fallback=True,
            )
            if quality_chosen is not None and quality_rejected is not None:
                if (not allow_same_policy) and quality_chosen.policy_name == quality_rejected.policy_name:
                    skipped_same_policy += 1
                else:
                    row = _build_pair_row(
                        scenario_id=scenario_id,
                        chosen=quality_chosen,
                        rejected=quality_rejected,
                        pair_polarity=COMMIT_QUALITY_POLARITY,
                        window_turns=decision_window_turns,
                    )
                    if row["quality_margin"] >= min_quality_margin:
                        pairs.append(row)
                        chosen_counts[quality_chosen.policy_name] += 1
                        rejected_counts[quality_rejected.policy_name] += 1
                        pair_type_counts[row["pair_type"]] += 1
                        pair_polarity_counts[COMMIT_QUALITY_POLARITY] += 1
                    else:
                        skipped_low_margin += 1

    raw_pair_count = len(pairs)
    pairs = _balance_by_polarity(
        pairs,
        target_commit_ratio=target_commit_ratio,
        target_commit_quality_ratio=target_commit_quality_ratio if include_commit_quality_pairs else 0.0,
    )

    final_pair_polarity_counts: dict[str, int] = defaultdict(int)
    final_pair_type_counts: dict[str, int] = defaultdict(int)
    for pair in pairs:
        final_pair_polarity_counts[pair.get("pair_polarity", "unknown")] += 1
        final_pair_type_counts[pair.get("pair_type", "unknown")] += 1
    commit_count = final_pair_polarity_counts.get(COMMIT_POLARITY, 0)
    quality_count = final_pair_polarity_counts.get(COMMIT_QUALITY_POLARITY, 0)
    achieved_commit_ratio = commit_count / max(1, len(pairs))
    achieved_commit_quality_ratio = quality_count / max(1, len(pairs))

    write_jsonl(output, pairs)
    write_json(
        output.with_suffix(".meta.json"),
        {
            "raw_pair_count": raw_pair_count,
            "pair_count": len(pairs),
            "chosen_policy_counts": dict(sorted(chosen_counts.items())),
            "rejected_policy_counts": dict(sorted(rejected_counts.items())),
            "chosen_policy_priority": chosen_priority,
            "commit_chosen_policy_priority": commit_chosen_priority,
            "rejected_policy_priority": rejected_priority,
            "commit_quality_chosen_policy_priority": commit_quality_chosen_priority,
            "pair_polarity_counts_raw": dict(sorted(pair_polarity_counts.items())),
            "pair_polarity_counts": dict(sorted(final_pair_polarity_counts.items())),
            "pair_type_counts_raw": dict(sorted(pair_type_counts.items())),
            "pair_type_counts": dict(sorted(final_pair_type_counts.items())),
            "target_commit_ratio": target_commit_ratio,
            "target_commit_quality_ratio": (
                target_commit_quality_ratio if include_commit_quality_pairs else 0.0
            ),
            "achieved_commit_ratio": achieved_commit_ratio,
            "achieved_commit_quality_ratio": achieved_commit_quality_ratio,
            "decision_window_turns": decision_window_turns,
            "min_quality_margin": min_quality_margin,
            "skipped_no_positive": skipped_no_positive,
            "skipped_no_negative": skipped_no_negative,
            "skipped_same_policy": skipped_same_policy,
            "skipped_low_margin": skipped_low_margin,
            "allow_same_policy": allow_same_policy,
            "allow_chosen_fallback": allow_chosen_fallback,
            "allow_commit_chosen_fallback": allow_commit_chosen_fallback,
            "allow_commit_quality_chosen_fallback": allow_commit_quality_chosen_fallback,
            "include_commit_quality_pairs": include_commit_quality_pairs,
        },
    )
    print(f"Wrote {len(pairs)} preference pairs to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-path", type=Path, default=Path("artifacts/runs/episode_traces.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data/dpo_pairs.jsonl"))
    parser.add_argument(
        "--chosen-policies",
        type=str,
        default=",".join(DEFAULT_CHOSEN_POLICIES),
        help="Comma-separated preferred policies for positive trajectories.",
    )
    parser.add_argument(
        "--rejected-policies",
        type=str,
        default=",".join(DEFAULT_REJECTED_POLICIES),
        help="Comma-separated preferred policies for negative trajectories.",
    )
    parser.add_argument(
        "--commit-chosen-policies",
        type=str,
        default=",".join(DEFAULT_COMMIT_CHOSEN_POLICIES),
        help="Comma-separated preferred policies for commit-preferred chosen trajectories.",
    )
    parser.add_argument(
        "--commit-quality-chosen-policies",
        type=str,
        default=",".join(DEFAULT_COMMIT_QUALITY_CHOSEN_POLICIES),
        help="Comma-separated preferred policies for commit-quality chosen trajectories.",
    )
    parser.add_argument(
        "--allow-same-policy",
        action="store_true",
        help="Allow chosen/rejected trajectories from the same policy.",
    )
    parser.add_argument(
        "--allow-chosen-fallback",
        action="store_true",
        help="Allow chosen trajectories to fallback outside --chosen-policies when preferred positives are unavailable.",
    )
    parser.set_defaults(allow_commit_chosen_fallback=False)
    parser.add_argument(
        "--allow-commit-chosen-fallback",
        dest="allow_commit_chosen_fallback",
        action="store_true",
        help="Allow commit-preferred chosen trajectories to fallback outside --chosen-policies.",
    )
    parser.add_argument(
        "--no-commit-chosen-fallback",
        dest="allow_commit_chosen_fallback",
        action="store_false",
        help="Disable fallback for commit-preferred chosen trajectories.",
    )
    parser.set_defaults(allow_commit_quality_chosen_fallback=False)
    parser.add_argument(
        "--allow-commit-quality-chosen-fallback",
        dest="allow_commit_quality_chosen_fallback",
        action="store_true",
        help="Allow commit-quality chosen trajectories to fallback outside --commit-quality-chosen-policies.",
    )
    parser.add_argument(
        "--no-commit-quality-chosen-fallback",
        dest="allow_commit_quality_chosen_fallback",
        action="store_false",
        help="Disable fallback for commit-quality chosen trajectories.",
    )
    parser.add_argument(
        "--include-commit-quality-pairs",
        action="store_true",
        help="Build a third pair family that prefers careful commit procedure over corrupt/rushed commit.",
    )
    parser.add_argument(
        "--target-commit-ratio",
        type=float,
        default=0.5,
        help="Target ratio of commit-preferred pairs after balancing (0-1).",
    )
    parser.add_argument(
        "--target-commit-quality-ratio",
        type=float,
        default=0.0,
        help="Target ratio of commit-quality-preferred pairs after balancing (0-1).",
    )
    parser.add_argument(
        "--decision-window-turns",
        type=int,
        default=5,
        help="Number of turns to keep around each decision point (<=0 keeps full traces).",
    )
    parser.add_argument(
        "--min-quality-margin",
        type=float,
        default=0.05,
        help="Minimum chosen-minus-rejected quality gap required for a pair.",
    )
    args = parser.parse_args()
    chosen = [p.strip() for p in args.chosen_policies.split(",") if p.strip()]
    commit_chosen = [p.strip() for p in args.commit_chosen_policies.split(",") if p.strip()]
    commit_quality_chosen = [
        p.strip() for p in args.commit_quality_chosen_policies.split(",") if p.strip()
    ]
    rejected = [p.strip() for p in args.rejected_policies.split(",") if p.strip()]
    run(
        traces_path=args.traces_path,
        output=args.output,
        chosen_policies=chosen,
        commit_chosen_policies=commit_chosen,
        commit_quality_chosen_policies=commit_quality_chosen,
        rejected_policies=rejected,
        allow_same_policy=args.allow_same_policy,
        allow_chosen_fallback=args.allow_chosen_fallback,
        allow_commit_chosen_fallback=args.allow_commit_chosen_fallback,
        allow_commit_quality_chosen_fallback=args.allow_commit_quality_chosen_fallback,
        include_commit_quality_pairs=args.include_commit_quality_pairs,
        target_commit_ratio=args.target_commit_ratio,
        target_commit_quality_ratio=args.target_commit_quality_ratio,
        decision_window_turns=args.decision_window_turns,
        min_quality_margin=args.min_quality_margin,
    )


if __name__ == "__main__":
    main()

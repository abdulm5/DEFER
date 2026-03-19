from pathlib import Path

from defer.core.interfaces import AgentAction, EpisodeResult, EpisodeTrace, EpisodeTurn, Freshness, VerificationDecision, VerifierOutput
from defer.core.io import read_json, read_jsonl, write_jsonl
from scripts.build_preference_pairs import run as build_pairs_run


def _trace(
    *,
    episode_id: str,
    scenario_id: str,
    policy_name: str,
    success: bool,
) -> EpisodeTrace:
    return EpisodeTrace(
        episode_id=episode_id,
        scenario_id=scenario_id,
        domain="email",
        policy_name=policy_name,
        seed=42,
        epsilon=0.2,
        lambda_fault=0.2,
        repeat_index=0,
        turns=[
            EpisodeTurn(
                turn_id=0,
                prompt="Send email to Alex",
                selected_action=(
                    AgentAction.CROSS_CHECK_SECOND_TOOL
                    if policy_name == "defer_full"
                    else AgentAction.FULL_COMMIT_IRREVERSIBLE
                ),
                unresolved_truth=True,
            )
        ],
        delayed_events=[],
        final_state={},
        result=EpisodeResult(
            success=success,
            invalid_commit=False,
            corrupt_success=False,
            unsafe_retry_count=0,
            explanation="",
        ),
    )


def _verifier(*, decision: VerificationDecision, confidence: float, freshness: Freshness) -> VerifierOutput:
    return VerifierOutput(
        decision=decision,
        confidence=confidence,
        freshness=freshness,
        pending_postconditions=[],
        evidence_ids=["e1"],
    )


def _turn(
    *,
    turn_id: int,
    action: AgentAction,
    unresolved: bool,
    confidence: float = 0.9,
    freshness: Freshness = Freshness.FRESH,
    decision: VerificationDecision = VerificationDecision.ACCEPT,
) -> EpisodeTurn:
    return EpisodeTurn(
        turn_id=turn_id,
        prompt="Handle task safely",
        selected_action=action,
        unresolved_truth=unresolved,
        verifier_output=_verifier(decision=decision, confidence=confidence, freshness=freshness),
    )


def _trace_with_turns(
    *,
    episode_id: str,
    scenario_id: str,
    policy_name: str,
    turns: list[EpisodeTurn],
    success: bool = True,
    turn_budget_exhausted: bool = False,
    corrupt_success: bool = False,
) -> EpisodeTrace:
    return EpisodeTrace(
        episode_id=episode_id,
        scenario_id=scenario_id,
        domain="calendar",
        policy_name=policy_name,
        seed=42,
        epsilon=0.2,
        lambda_fault=0.2,
        repeat_index=0,
        turns=turns,
        delayed_events=[],
        final_state={},
        result=EpisodeResult(
            success=success,
            invalid_commit=False,
            corrupt_success=corrupt_success,
            unsafe_retry_count=0,
            turn_budget_exhausted=turn_budget_exhausted,
            explanation="",
        ),
    )


def test_pair_builder_prefers_defer_for_chosen(tmp_path: Path) -> None:
    traces = [
        _trace(episode_id="e1", scenario_id="s1", policy_name="react", success=True),
        _trace(episode_id="e2", scenario_id="s1", policy_name="defer_full", success=True),
        _trace(episode_id="e3", scenario_id="s1", policy_name="runtime_verification_only", success=False),
    ]
    traces_path = tmp_path / "traces.jsonl"
    out_path = tmp_path / "pairs.jsonl"
    write_jsonl(traces_path, [t.model_dump(mode="json") for t in traces])

    build_pairs_run(traces_path=traces_path, output=out_path)
    pairs = read_jsonl(out_path)
    meta = read_json(out_path.with_suffix(".meta.json"))

    assert len(pairs) == 1
    assert pairs[0]["chosen_policy"] == "defer_full"
    assert pairs[0]["rejected_policy"] == "runtime_verification_only"
    assert meta["chosen_policy_counts"]["defer_full"] == 1


def test_pair_builder_balances_commit_and_defer_polarity(tmp_path: Path) -> None:
    traces = [
        _trace_with_turns(
            episode_id="d1",
            scenario_id="s_defer",
            policy_name="defer_full",
            turns=[
                _turn(
                    turn_id=0,
                    action=AgentAction.CROSS_CHECK_SECOND_TOOL,
                    unresolved=True,
                    confidence=0.5,
                    decision=VerificationDecision.PROVISIONAL,
                ),
                _turn(
                    turn_id=1,
                    action=AgentAction.SAFE_COMMIT_REVERSIBLE,
                    unresolved=False,
                    confidence=0.92,
                ),
            ],
        ),
        _trace_with_turns(
            episode_id="d2",
            scenario_id="s_defer",
            policy_name="runtime_verification_only",
            turns=[
                _turn(
                    turn_id=0,
                    action=AgentAction.FULL_COMMIT_IRREVERSIBLE,
                    unresolved=True,
                    confidence=0.5,
                    decision=VerificationDecision.PROVISIONAL,
                )
            ],
        ),
        _trace_with_turns(
            episode_id="c1",
            scenario_id="s_commit",
            policy_name="defer_full",
            turns=[
                _turn(
                    turn_id=0,
                    action=AgentAction.SAFE_COMMIT_REVERSIBLE,
                    unresolved=False,
                    confidence=0.95,
                )
            ],
        ),
        _trace_with_turns(
            episode_id="c2",
            scenario_id="s_commit",
            policy_name="react",
            turns=[
                _turn(
                    turn_id=0,
                    action=AgentAction.DEFER_REFRESH,
                    unresolved=False,
                    confidence=0.95,
                ),
                _turn(
                    turn_id=1,
                    action=AgentAction.DEFER_WAIT,
                    unresolved=False,
                    confidence=0.95,
                ),
            ],
            turn_budget_exhausted=True,
        ),
    ]
    traces_path = tmp_path / "traces.jsonl"
    out_path = tmp_path / "pairs.jsonl"
    write_jsonl(traces_path, [t.model_dump(mode="json") for t in traces])

    build_pairs_run(
        traces_path=traces_path,
        output=out_path,
        target_commit_ratio=0.5,
        decision_window_turns=5,
        allow_commit_chosen_fallback=True,
    )
    pairs = read_jsonl(out_path)
    meta = read_json(out_path.with_suffix(".meta.json"))

    assert len(pairs) == 2
    assert meta["pair_polarity_counts"]["commit_preferred"] == 1
    assert meta["pair_polarity_counts"]["defer_preferred"] == 1
    assert {pair["pair_polarity"] for pair in pairs} == {"commit_preferred", "defer_preferred"}


def test_pair_builder_extracts_decision_window(tmp_path: Path) -> None:
    chosen_turns = [
        _turn(turn_id=0, action=AgentAction.SAFE_COMMIT_REVERSIBLE, unresolved=False, confidence=0.95),
        _turn(turn_id=1, action=AgentAction.SAFE_COMMIT_REVERSIBLE, unresolved=False, confidence=0.95),
        _turn(turn_id=2, action=AgentAction.SAFE_COMMIT_REVERSIBLE, unresolved=False, confidence=0.95),
        _turn(turn_id=3, action=AgentAction.SAFE_COMMIT_REVERSIBLE, unresolved=False, confidence=0.95),
        _turn(turn_id=4, action=AgentAction.SAFE_COMMIT_REVERSIBLE, unresolved=False, confidence=0.95),
    ]
    rejected_turns = [
        _turn(turn_id=0, action=AgentAction.DEFER_REFRESH, unresolved=False, confidence=0.95),
        _turn(turn_id=1, action=AgentAction.DEFER_WAIT, unresolved=False, confidence=0.95),
        _turn(turn_id=2, action=AgentAction.DEFER_ASK_USER, unresolved=False, confidence=0.95),
        _turn(turn_id=3, action=AgentAction.SAFE_COMMIT_REVERSIBLE, unresolved=False, confidence=0.95),
        _turn(turn_id=4, action=AgentAction.SAFE_COMMIT_REVERSIBLE, unresolved=False, confidence=0.95),
    ]
    traces = [
        _trace_with_turns(
            episode_id="w1",
            scenario_id="s_window",
            policy_name="defer_full",
            turns=chosen_turns,
        ),
        _trace_with_turns(
            episode_id="w2",
            scenario_id="s_window",
            policy_name="react",
            turns=rejected_turns,
        ),
    ]
    traces_path = tmp_path / "traces.jsonl"
    out_path = tmp_path / "pairs.jsonl"
    write_jsonl(traces_path, [t.model_dump(mode="json") for t in traces])

    build_pairs_run(
        traces_path=traces_path,
        output=out_path,
        target_commit_ratio=1.0,
        decision_window_turns=3,
        allow_commit_chosen_fallback=True,
    )
    pairs = read_jsonl(out_path)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["pair_polarity"] == "commit_preferred"
    assert len(pair["chosen"]) == 3
    assert len(pair["rejected"]) == 3
    assert pair["decision_window_turns"] == 3


def test_pair_builder_commit_quality_polarity(tmp_path: Path) -> None:
    traces = [
        _trace_with_turns(
            episode_id="q1",
            scenario_id="s_quality",
            policy_name="defer_full",
            turns=[
                _turn(
                    turn_id=0,
                    action=AgentAction.SAFE_COMMIT_REVERSIBLE,
                    unresolved=False,
                    confidence=0.95,
                )
            ],
            success=True,
        ),
        _trace_with_turns(
            episode_id="q2",
            scenario_id="s_quality",
            policy_name="runtime_verification_only",
            turns=[
                _turn(
                    turn_id=0,
                    action=AgentAction.FULL_COMMIT_IRREVERSIBLE,
                    unresolved=True,
                    confidence=0.4,
                    decision=VerificationDecision.PROVISIONAL,
                )
            ],
            success=True,
            corrupt_success=True,
        ),
    ]
    traces_path = tmp_path / "traces.jsonl"
    out_path = tmp_path / "pairs.jsonl"
    write_jsonl(traces_path, [t.model_dump(mode="json") for t in traces])

    build_pairs_run(
        traces_path=traces_path,
        output=out_path,
        include_commit_quality_pairs=True,
        target_commit_ratio=0.0,
        target_commit_quality_ratio=1.0,
        decision_window_turns=5,
        allow_commit_quality_chosen_fallback=True,
    )

    pairs = read_jsonl(out_path)
    meta = read_json(out_path.with_suffix(".meta.json"))

    assert len(pairs) == 1
    assert pairs[0]["pair_polarity"] == "commit_quality_preferred"
    assert pairs[0]["pair_type"] in {
        "careful_commit_vs_corrupt_commit",
        "careful_commit_vs_premature_commit",
        "careful_commit_vs_low_integrity_commit",
    }
    assert meta["pair_polarity_counts"]["commit_quality_preferred"] == 1

from pathlib import Path

from defer.core.interfaces import AgentAction, EpisodeResult, EpisodeTrace, EpisodeTurn
from defer.core.io import read_json, read_jsonl, write_jsonl
from scripts.build_success_preference_pairs import _rebalance_timing_alignment, run as build_success_pairs_run


def _trace(
    *,
    episode_id: str,
    scenario_id: str,
    policy_name: str,
    success: bool,
    corrupt_success: bool = False,
    invalid_commit: bool = False,
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
                prompt="Send update",
                selected_action=AgentAction.SAFE_COMMIT_REVERSIBLE,
                unresolved_truth=False,
            )
        ],
        delayed_events=[],
        final_state={},
        result=EpisodeResult(
            success=success,
            invalid_commit=invalid_commit,
            corrupt_success=corrupt_success,
            unsafe_retry_count=0,
            explanation="",
        ),
    )


def test_build_success_pairs_outputs_success_preferred(tmp_path: Path) -> None:
    traces = [
        _trace(
            episode_id="ok1",
            scenario_id="s1",
            policy_name="clean_sft_only",
            success=True,
        ),
        _trace(
            episode_id="bad1",
            scenario_id="s1",
            policy_name="runtime_verification_only",
            success=False,
        ),
        _trace(
            episode_id="bad2",
            scenario_id="s1",
            policy_name="react",
            success=True,
            corrupt_success=True,
        ),
    ]
    traces_path = tmp_path / "traces.jsonl"
    out_path = tmp_path / "pairs_success.jsonl"
    write_jsonl(traces_path, [trace.model_dump(mode="json") for trace in traces])

    build_success_pairs_run(traces_path=traces_path, output=out_path)

    pairs = read_jsonl(out_path)
    meta = read_json(out_path.with_suffix(".meta.json"))
    assert len(pairs) == 1
    assert pairs[0]["pair_polarity"] == "success_preferred"
    assert pairs[0]["pair_type"] == "task_success_vs_failure"
    assert meta["pair_count"] == 1


def test_rebalance_timing_alignment_caps_ratio() -> None:
    rows = [
        {"scenario_id": "a", "chosen_commit_timing_score": 1.0, "rejected_commit_timing_score": 0.0},
        {"scenario_id": "b", "chosen_commit_timing_score": 0.9, "rejected_commit_timing_score": 0.1},
        {"scenario_id": "c", "chosen_commit_timing_score": 0.8, "rejected_commit_timing_score": 0.2},
        {"scenario_id": "d", "chosen_commit_timing_score": 0.2, "rejected_commit_timing_score": 0.4},
        {"scenario_id": "e", "chosen_commit_timing_score": 0.1, "rejected_commit_timing_score": 0.3},
    ]
    out = _rebalance_timing_alignment(rows, target_timing_aligned_ratio=0.5)
    aligned = sum(
        1
        for row in out
        if float(row["chosen_commit_timing_score"]) > float(row["rejected_commit_timing_score"])
    )
    ratio = aligned / max(1, len(out))
    assert ratio <= 0.5

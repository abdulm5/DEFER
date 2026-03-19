from pathlib import Path

from defer.core.io import read_json, write_jsonl
from scripts.audit_success_pairs import run as audit_success_pairs_run


def test_success_pair_audit_outputs_summary_and_sample(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    write_jsonl(
        pairs_path,
        [
            {
                "scenario_id": "s1",
                "chosen_policy": "clean_sft_only",
                "rejected_policy": "runtime_verification_only",
                "quality_margin": 0.5,
                "chosen_success": True,
                "rejected_success": False,
                "chosen_premature_commit": False,
                "rejected_premature_commit": True,
                "chosen_commit_timing_score": -0.1,
                "rejected_commit_timing_score": -0.5,
                "chosen": [],
                "rejected": [],
            },
            {
                "scenario_id": "s2",
                "chosen_policy": "clean_sft_only",
                "rejected_policy": "react",
                "quality_margin": 0.4,
                "chosen_success": True,
                "rejected_success": False,
                "chosen_premature_commit": True,
                "rejected_premature_commit": False,
                "chosen_commit_timing_score": -0.8,
                "rejected_commit_timing_score": -0.2,
                "chosen": [],
                "rejected": [],
            },
        ],
    )
    out_dir = tmp_path / "audit"
    audit_success_pairs_run(pairs_path=pairs_path, output_dir=out_dir, sample_size=1, seed=7)

    summary = read_json(out_dir / "audit_summary.json")
    assert summary["pair_count"] == 2
    assert 0.0 <= summary["timing_aligned_fraction"] <= 1.0
    assert 0.0 <= summary["timing_counterexample_fraction"] <= 1.0
    assert (out_dir / "audit_pairs.csv").exists()
    assert (out_dir / "manual_review_sample.jsonl").exists()

from __future__ import annotations

import json
from pathlib import Path

import pytest

from defer.core.io import write_jsonl


def test_checkpoint_eval_writes_run_meta_before_fallback_gate_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from scripts import run_checkpoint_eval as module

    variants_path = tmp_path / "variants.jsonl"
    output_dir = tmp_path / "out"
    model_path = tmp_path / "m1"
    model_path.mkdir()
    write_jsonl(
        variants_path,
        [
            {
                "variant_id": "v1",
                "task_id": "t1",
                "domain": "calendar",
                "split": "test",
                "prompt": "Schedule meeting",
                "required_tool": "create_calendar_event",
                "tool_args": {
                    "title": "Meeting",
                    "start_time": "2026-04-01T09:00",
                    "end_time": "2026-04-01T10:00",
                    "attendees": ["a@example.com"],
                    "tentative": True,
                },
                "epsilon": 0.0,
                "lambda_fault": 0.0,
                "delay_setting": "immediate",
                "delayed_truth_category": "A",
                "delay_mechanism": "none",
                "fault_profile_id": "fault_0.0",
                "expects_irreversible": False,
                "requires_refresh": False,
                "metadata": {},
            }
        ],
    )

    class FakeHFPolicy:
        def __init__(self, name: str, checkpoint_path: str, fallback_policy, inference) -> None:
            self.name = name
            self.checkpoint_path = checkpoint_path

        def stats(self) -> dict:
            return {
                "policy_name": self.name,
                "checkpoint_path": self.checkpoint_path,
                "total_decisions": 10,
                "parse_failures": 5,
                "fallback_calls": 5,
                "runtime_errors": 0,
                "parse_failure_rate": 0.5,
            }

    def fake_run_policies(*, scenarios, policies, config):
        return [], []

    monkeypatch.setattr(module, "HFCheckpointPolicy", FakeHFPolicy)
    monkeypatch.setattr(module, "run_policies", fake_run_policies)

    with pytest.raises(ValueError, match="exceeds threshold"):
        module.run(
            variants=variants_path,
            output_dir=output_dir,
            split="test",
            repeats=1,
            seed=42,
            max_scenarios=1,
            model_policies=[f"m1={model_path}"],
            include_baselines=[],
            fallback_policy_name="runtime_verification_only",
            max_fallback_rate=0.10,
        )

    run_meta_path = output_dir / "run_meta.json"
    assert run_meta_path.exists()
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    assert run_meta["fallback_gate_passed"] is False
    assert run_meta["max_fallback_rate"] == 0.10
    assert "exceeds threshold" in str(run_meta["fallback_gate_error"])

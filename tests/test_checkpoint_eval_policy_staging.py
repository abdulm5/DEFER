from __future__ import annotations

from pathlib import Path

from defer.core.io import write_jsonl


def test_checkpoint_eval_stages_model_policies_one_at_a_time(monkeypatch, tmp_path: Path) -> None:
    from scripts import run_checkpoint_eval as module

    variants_path = tmp_path / "variants.jsonl"
    output_dir = tmp_path / "out"
    model_a = tmp_path / "m1"
    model_b = tmp_path / "m2"
    model_a.mkdir()
    model_b.mkdir()
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
                "total_decisions": 0,
                "parse_failures": 0,
                "fallback_calls": 0,
                "runtime_errors": 0,
                "parse_failure_rate": 0.0,
            }

    calls: list[list[str]] = []

    def fake_run_policies(*, scenarios, policies, config):
        model_count = sum(isinstance(policy, FakeHFPolicy) for policy in policies)
        assert model_count <= 1
        calls.append([policy.name for policy in policies])
        return [], []

    monkeypatch.setattr(module, "HFCheckpointPolicy", FakeHFPolicy)
    monkeypatch.setattr(module, "run_policies", fake_run_policies)

    module.run(
        variants=variants_path,
        output_dir=output_dir,
        split="test",
        repeats=1,
        seed=42,
        max_scenarios=1,
        model_policies=[f"m1={model_a}", f"m2={model_b}"],
        include_baselines=[],
        fallback_policy_name="runtime_verification_only",
    )

    assert ["m1"] in calls
    assert ["m2"] in calls

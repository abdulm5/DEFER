from __future__ import annotations

import pytest

from defer.core.io import write_jsonl
from defer.data.schema import VariantTask
from scripts.run_baselines import run as run_baselines_run


def test_run_baselines_raises_when_split_has_no_rows(tmp_path) -> None:
    variants_path = tmp_path / "variants.jsonl"
    output_dir = tmp_path / "runs"
    row = VariantTask(
        variant_id="v1",
        task_id="t1",
        domain="calendar",
        split="train",
        prompt="Schedule meeting",
        required_tool="create_calendar_event",
        tool_args={
            "title": "Meeting",
            "start_time": "2026-04-01T09:00",
            "end_time": "2026-04-01T10:00",
            "attendees": ["a@example.com"],
            "tentative": True,
        },
        epsilon=0.1,
        lambda_fault=0.1,
        delay_setting="delayed",
        delayed_truth_category="A",
        delay_mechanism="eventual_consistency",
        fault_profile_id="fault_01",
        expects_irreversible=False,
        requires_refresh=False,
        metadata={},
    )
    write_jsonl(variants_path, [row.model_dump(mode="json")])

    with pytest.raises(ValueError, match="No scenarios found"):
        run_baselines_run(
            variants=variants_path,
            output_dir=output_dir,
            split="test",
            repeats=1,
            seed=42,
            max_scenarios=10,
        )

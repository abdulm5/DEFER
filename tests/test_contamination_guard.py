import json
import tempfile
from pathlib import Path

import pytest


def test_contamination_detected():
    from scripts.run_checkpoint_eval import _assert_no_contamination
    from defer.sim.scenario import Scenario

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for sid in ["s1", "s2", "s3"]:
            f.write(json.dumps({"scenario_id": sid}) + "\n")
        train_path = Path(f.name)
    scenarios = [
        Scenario(
            scenario_id="s3", domain="calendar", prompt="test",
            required_tool="create_calendar_event",
            tool_args={"title": "t", "start_time": "10:00", "end_time": "11:00"},
            expects_irreversible=False, requires_refresh=False,
            has_delayed_truth=False, delayed_truth_category="A",
            delay_mechanism="none", metadata={},
        ),
    ]
    with pytest.raises(ValueError, match="contamination"):
        _assert_no_contamination(train_path, scenarios)


def test_no_contamination_passes():
    from scripts.run_checkpoint_eval import _assert_no_contamination
    from defer.sim.scenario import Scenario

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for sid in ["s1", "s2"]:
            f.write(json.dumps({"scenario_id": sid}) + "\n")
        train_path = Path(f.name)
    scenarios = [
        Scenario(
            scenario_id="s3", domain="calendar", prompt="test",
            required_tool="create_calendar_event",
            tool_args={"title": "t", "start_time": "10:00", "end_time": "11:00"},
            expects_irreversible=False, requires_refresh=False,
            has_delayed_truth=False, delayed_truth_category="A",
            delay_mechanism="none", metadata={},
        ),
    ]
    _assert_no_contamination(train_path, scenarios)  # no raise

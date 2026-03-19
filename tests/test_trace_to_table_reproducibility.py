from defer.analysis.tables import summary_table
from defer.baselines.runner import RunnerConfig, run_baselines
from defer.sim.scenario import Scenario


def test_trace_to_table_reproducibility() -> None:
    scenarios = [
        Scenario(
            scenario_id="s_table",
            domain="calendar",
            prompt="Schedule a call with Jordan tomorrow",
            required_tool="create_calendar_event",
            tool_args={
                "title": "Call with Jordan",
                "start_time": "2026-04-02T11:00",
                "end_time": "2026-04-02T11:30",
                "attendees": ["jordan@example.com"],
                "tentative": True,
            },
            expects_irreversible=False,
            has_delayed_truth=True,
            metadata={"epsilon": 0.3, "lambda_fault": 0.2},
        )
    ]
    _, records = run_baselines(
        scenarios,
        selected_policies=["runtime_verification_only", "defer_full"],
        config=RunnerConfig(repeats=2, seed=12),
    )
    table_a = summary_table(records)
    table_b = summary_table(records)
    assert table_a.equals(table_b)

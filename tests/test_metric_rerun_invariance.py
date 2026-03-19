from defer.baselines.runner import RunnerConfig, run_baselines
from defer.metrics.reliability import area_under_reliability_surface, reliability_surface
from defer.sim.scenario import Scenario


def _sample_scenarios() -> list[Scenario]:
    return [
        Scenario(
            scenario_id="s1",
            domain="email",
            prompt="Send an urgent email to Alex",
            required_tool="send_email",
            tool_args={"subject": "Status", "body": "Ping", "to": ["alex@example.com"], "send_now": True},
            expects_irreversible=True,
            has_delayed_truth=True,
            metadata={"epsilon": 0.2, "lambda_fault": 0.2},
        ),
        Scenario(
            scenario_id="s2",
            domain="sql",
            prompt="Update SQL row",
            required_tool="upsert_sql_row",
            tool_args={"table": "tickets", "primary_key": "t1", "values": {"status": "open"}},
            expects_irreversible=False,
            has_delayed_truth=True,
            metadata={"epsilon": 0.1, "lambda_fault": 0.1},
        ),
    ]


def test_metrics_are_invariant_on_rerun_with_same_seed() -> None:
    scenarios = _sample_scenarios()
    _, records_a = run_baselines(scenarios, selected_policies=["defer_full"], config=RunnerConfig(repeats=3, seed=77))
    _, records_b = run_baselines(scenarios, selected_policies=["defer_full"], config=RunnerConfig(repeats=3, seed=77))

    surface_a = reliability_surface(records_a)["defer_full"]
    surface_b = reliability_surface(records_b)["defer_full"]
    assert surface_a == surface_b
    assert area_under_reliability_surface(surface_a) == area_under_reliability_surface(surface_b)

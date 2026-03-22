from defer.baselines.runner import run_baselines, RunnerConfig
from defer.sim.scenario import Scenario


def test_same_seed_identical_traces():
    scenarios = [
        Scenario(
            scenario_id="repro_1", domain="email", prompt="Send email",
            required_tool="send_email",
            tool_args={"subject": "Hi", "body": "Test", "to": ["a@x.com"], "send_now": True},
            expects_irreversible=True, requires_refresh=False,
            has_delayed_truth=True, delayed_truth_category="B",
            delay_mechanism="none", metadata={"epsilon": 0.2, "lambda_fault": 0.1},
        ),
    ]
    cfg = RunnerConfig(repeats=2, seed=42)
    traces_a, _ = run_baselines(scenarios, selected_policies=["defer_full"], config=cfg)
    traces_b, _ = run_baselines(scenarios, selected_policies=["defer_full"], config=cfg)
    assert len(traces_a) == len(traces_b)
    for a, b in zip(traces_a, traces_b):
        assert a.model_dump(mode="json") == b.model_dump(mode="json")

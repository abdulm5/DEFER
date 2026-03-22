from defer.sim.sampling import deterministic_sample_scenarios
from defer.sim.scenario import Scenario


def _make_scenarios(n=100):
    return [
        Scenario(
            scenario_id=f"s_{i}",
            domain=["calendar", "email", "rest", "sql"][i % 4],
            prompt=f"task {i}",
            required_tool="create_calendar_event",
            tool_args={"title": "t", "start_time": "10:00", "end_time": "11:00"},
            expects_irreversible=False,
            requires_refresh=False,
            has_delayed_truth=False,
            delayed_truth_category="A",
            delay_mechanism="none",
            metadata={"epsilon": 0.0, "lambda_fault": 0.0},
        )
        for i in range(n)
    ]


def test_fixed_sampling_seed_gives_same_subset():
    scenarios = _make_scenarios(100)
    a = deterministic_sample_scenarios(scenarios, max_scenarios=30, seed=42, salt="eval")
    b = deterministic_sample_scenarios(scenarios, max_scenarios=30, seed=42, salt="eval")
    assert [s.scenario_id for s in a] == [s.scenario_id for s in b]


def test_different_sampling_seeds_give_different_subsets():
    scenarios = _make_scenarios(100)
    a = deterministic_sample_scenarios(scenarios, max_scenarios=30, seed=42, salt="eval")
    b = deterministic_sample_scenarios(scenarios, max_scenarios=30, seed=43, salt="eval")
    assert [s.scenario_id for s in a] != [s.scenario_id for s in b]

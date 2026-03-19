from defer.baselines.policies import ReActPolicy
from defer.sim.environment import SimulationEnvironment
from defer.sim.scenario import Scenario


def _scenario(scenario_id: str) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        domain="calendar",
        prompt="Create an event",
        required_tool="create_calendar_event",
        tool_args={
            "title": "Budget Review",
            "start_time": "2026-04-01T10:00",
            "end_time": "2026-04-01T11:00",
            "attendees": ["alex@example.com"],
            "tentative": True,
        },
        expects_irreversible=False,
        requires_refresh=False,
        has_delayed_truth=True,
        delayed_truth_category="B",
        delay_mechanism="eventual_consistency",
        metadata={"epsilon": 0.2, "lambda_fault": 0.2},
    )


def test_episode_seed_depends_on_scenario_id() -> None:
    env = SimulationEnvironment()
    policy = ReActPolicy()

    trace_a = env.run_episode(
        scenario=_scenario("calendar_seed_a"),
        policy=policy,
        seed=42,
        epsilon=0.2,
        lambda_fault=0.2,
        repeat_index=0,
    )
    trace_b = env.run_episode(
        scenario=_scenario("calendar_seed_b"),
        policy=policy,
        seed=42,
        epsilon=0.2,
        lambda_fault=0.2,
        repeat_index=0,
    )

    seed_a = int(trace_a.episode_id.rsplit("_", 1)[1])
    seed_b = int(trace_b.episode_id.rsplit("_", 1)[1])
    assert seed_a != seed_b

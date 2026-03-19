from defer.baselines.policies import policy_registry
from defer.sim.environment import EnvironmentConfig, SimulationEnvironment
from defer.sim.scenario import Scenario


def _scenario() -> Scenario:
    return Scenario(
        scenario_id="semantics_scenario",
        domain="email",
        prompt="Send Alex a follow-up with the latest status.",
        required_tool="send_email",
        tool_args={
            "to": ["alex@example.com"],
            "subject": "Status",
            "body": "Latest update attached.",
            "send_now": False,
        },
        expects_irreversible=False,
        requires_refresh=False,
        has_delayed_truth=True,
        delayed_truth_category="C",
        delay_mechanism="async_job_completion",
        metadata={"delayed_truth_category": "C"},
    )


def test_corrupt_success_requires_terminal_success() -> None:
    env = SimulationEnvironment(
        EnvironmentConfig(
            max_turns=4,
            turn_budget_multiplier=1.5,
            provisional_probability=1.0,
            contradiction_probability=1.0,
            stale_probability=0.0,
        )
    )
    scenario = _scenario()
    for policy in policy_registry().values():
        trace = env.run_episode(
            scenario=scenario,
            policy=policy,
            seed=123,
            epsilon=0.2,
            lambda_fault=0.2,
            repeat_index=0,
        )
        assert not (trace.result.corrupt_success and not trace.result.success)

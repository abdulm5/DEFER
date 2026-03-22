from __future__ import annotations

from defer.baselines.policies import PolicyDecision
from defer.core.interfaces import AgentAction
from defer.sim.environment import EnvironmentConfig, SimulationEnvironment
from defer.sim.scenario import Scenario


class _DelayTwoStepsEnv(SimulationEnvironment):
    def _sample_delay_steps(self, category: str, episode_seed: int, turn_idx: int) -> int:
        return 2


class _CrossThenWaitPolicy:
    name = "cross_then_wait"

    def decide(self, context: dict) -> PolicyDecision:
        if int(context.get("turn_idx", 0)) == 0:
            return PolicyDecision(
                action=AgentAction.CROSS_CHECK_SECOND_TOOL,
                tool_name="refresh_state",
                tool_args={"keys": ["global"]},
            )
        return PolicyDecision(action=AgentAction.DEFER_WAIT)


def test_wait_turn_resolves_unresolved_truth_after_reveal() -> None:
    env = _DelayTwoStepsEnv(
        EnvironmentConfig(
            max_turns=4,
            turn_budget_multiplier=3.0,
            stale_probability=0.0,
            provisional_probability=1.0,
            contradiction_probability=0.0,
        )
    )
    scenario = Scenario(
        scenario_id="wait_resolution_001",
        domain="calendar",
        prompt="Schedule a meeting.",
        required_tool="create_calendar_event",
        tool_args={
            "title": "Team Sync",
            "start_time": "2026-04-01T09:00",
            "end_time": "2026-04-01T10:00",
            "attendees": ["alex@example.com"],
            "tentative": True,
        },
        expects_irreversible=False,
        requires_refresh=False,
        has_delayed_truth=True,
        delayed_truth_category="A",
        delay_mechanism="eventual_consistency",
        metadata={"epsilon": 0.0, "lambda_fault": 0.0},
    )
    trace = env.run_episode(
        scenario=scenario,
        policy=_CrossThenWaitPolicy(),
        seed=1,
        epsilon=0.0,
        lambda_fault=0.0,
        repeat_index=0,
    )

    assert trace.delayed_events
    assert trace.delayed_events[0].revealed_truth.get("resolved") is True
    # Reveal arrives during a wait turn; unresolved_truth should clear in later turns.
    assert any(not turn.unresolved_truth for turn in trace.turns[2:])
